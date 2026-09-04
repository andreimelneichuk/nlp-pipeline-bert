import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================
# 1. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ
# =====================

# Загрузка данных
df = pd.read_csv('output/documentation_dataset_labeled.csv')  # Замените на ваш путь к файлу

# Первичный анализ
print("Размер датасета:", df.shape)
print("\nБаланс классов:")
print(df['category_name'].value_counts())
print("\nКоличество классов:", df['category_name'].nunique())

# Отбираем только нужные столбцы
data = df[['content', 'category_name']].copy()

# Лёгкая очистка текста (удаление URL, лишних пробелов)
import re
def clean_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data['content'] = data['content'].apply(clean_text)
print(f"\nВсего примеров: {len(data)}")

# =====================
# 2. ПОДГОТОВКА ДАННЫХ
# =====================

# Создаем маппинг категорий в числовые метки
unique_categories = data['category_name'].unique()
label2id = {label: idx for idx, label in enumerate(unique_categories)}
id2label = {idx: label for label, idx in label2id.items()}

data['label'] = data['category_name'].map(label2id)
print(f"Количество классов: {len(label2id)}")
print("Маппинг категорий:", label2id)

# Разделение на train/val/test
train_df, temp_df = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"\nРазмеры выборок:")
print(f"Обучающая: {len(train_df)}")
print(f"Валидационная: {len(val_df)}")
print(f"Тестовая: {len(test_df)}")

# =====================
# 3. НАСТРОЙКА ТОКЕНИЗАТОРА И DATASET
# =====================

MODEL_NAME = 'ai-forever/ruBert-base'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Создаем datasets
train_dataset = TextClassificationDataset(
    train_df['content'], train_df['label'], tokenizer
)
val_dataset = TextClassificationDataset(
    val_df['content'], val_df['label'], tokenizer
)
test_dataset = TextClassificationDataset(
    test_df['content'], test_df['label'], tokenizer
)

# Создаем DataLoaders
BATCH_SIZE = 16

# Oversampling редких классов с помощью WeightedRandomSampler
class_counts = train_df['label'].value_counts().to_dict()
class_weights = {label: 1.0 / count for label, count in class_counts.items()}
sample_weights = train_df['label'].map(class_weights).values
from torch.utils.data import WeightedRandomSampler
train_sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =====================
# 4. БАЗОВОЕ ОБУЧЕНИЕ
# =====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nИспользуется устройство: {device}")

# Загрузка модели
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.1,  # Добавили dropout для регуляризации
    attention_probs_dropout_prob=0.1
)
model.to(device)

# Настройка оптимизатора и шедулера с warmup=10%
LEARNING_RATE = 1e-5  # Уменьшили для более стабильного обучения
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)  # Добавили weight_decay
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Уменьшили для стабильности
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions, average='weighted')
    
    return accuracy, f1, predictions, actual_labels

# Обучение модели
print("\nНачало обучения...")
best_f1 = 0
epochs_without_improve = 0
PATIENCE = 3  # Увеличили patience для более стабильного обучения
for epoch in range(EPOCHS):
    print(f"\nЭпоха {epoch + 1}/{EPOCHS}")
    
    # Обучение
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Потери на обучении: {train_loss:.4f}")
    
    # Валидация
    val_accuracy, val_f1, _, _ = evaluate_model(model, val_loader, device)
    print(f"Валидационная Accuracy: {val_accuracy:.4f}")
    print(f"Валидационный F1-score: {val_f1:.4f}")
    
    # Сохраняем лучшую модель + ранняя остановка
    if val_f1 > best_f1:
        best_f1 = val_f1
        epochs_without_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improve += 1
        if epochs_without_improve > PATIENCE:
            print(f"Ранняя остановка: нет улучшений {PATIENCE} эпох подряд")
            break

# Загрузка лучшей модели для тестирования
model.load_state_dict(torch.load('best_model.pth'))

# Оценка на тестовой выборке
print("\nОценка на тестовой выборке...")
test_accuracy, test_f1, test_preds, test_labels = evaluate_model(model, test_loader, device)
print(f"Тестовая Accuracy: {test_accuracy:.4f}")
print(f"Тестовый F1-score: {test_f1:.4f}")

# =====================
# 5. КВАНТИЗАЦИЯ МОДЕЛИ
# =====================

print("\nПрименение квантизации...")

# Сохраняем оригинальный размер модели
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Оригинальный размер модели: {original_size / 1024 / 1024:.2f} MB")

# Замеряем скорость инференса до квантизации
start_time = time.time()
_, _, _, _ = evaluate_model(model, test_loader, device)
original_inference_time = time.time() - start_time
print(f"Оригинальное время инференса: {original_inference_time:.2f} сек")

quantization_failed = False
try:
    # Пытаемся включить доступный движок квантизации на CPU
    if getattr(torch.backends.quantized, 'engine', 'none') == 'none':
        try:
            torch.backends.quantized.engine = 'qnnpack'
        except Exception:
            pass

    # Применяем динамическую квантизацию
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Сохраняем квантизированный размер модели
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    print(f"Квантизированный размер модели: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"Сжатие: {original_size/quantized_size:.2f}x")

    # Замеряем скорость инференса после квантизации
    start_time = time.time()
    quantized_accuracy, quantized_f1, _, _ = evaluate_model(quantized_model, test_loader, device)
    quantized_inference_time = time.time() - start_time
    print(f"Квантизированное время инференса: {quantized_inference_time:.2f} сек")
    print(f"Ускорение: {original_inference_time/quantized_inference_time:.2f}x")
except Exception as e:
    # Квантизация недоступна в текущей сборке PyTorch (например, NoQEngine на macOS)
    print(f"Квантизация недоступна на этой сборке PyTorch: {e}")
    quantization_failed = True
    quantized_model = model
    quantized_size = original_size
    quantized_accuracy, quantized_f1 = test_accuracy, test_f1
    quantized_inference_time = original_inference_time

print(f"\nТочность после квантизации:")
print(f"Accuracy: {quantized_accuracy:.4f} (было {test_accuracy:.4f})")
print(f"F1-score: {quantized_f1:.4f} (было {test_f1:.4f})")

# =====================
# 6. ФИНАЛЬНАЯ ОЦЕНКА И ПРИМЕРЫ
# =====================

def predict_text(text, model, tokenizer, device, max_length=384):
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(outputs.logits, dim=1)
    
    predicted_label = model.config.id2label[prediction.item()]
    confidence = probabilities[0][prediction.item()].item()
    
    return predicted_label, confidence

print("\nПримеры работы модели:")
test_texts = test_df['content'].head(3).tolist()
true_labels = test_df['category_name'].head(3).tolist()

for i, (text, true_label) in enumerate(zip(test_texts, true_labels)):
    predicted_label, confidence = predict_text(text, quantized_model, tokenizer, device)
    print(f"\nПример {i+1}:")
    print(f"Текст: {text[:100]}...")
    print(f"Истинная метка: {true_label}")
    print(f"Предсказанная метка: {predicted_label}")
    print(f"Уверенность: {confidence:.4f}")
    print(f"Правильно: {true_label == predicted_label}")

# Сохраняем квантизированную модель
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
print("\nКвантизированная модель сохранена как 'quantized_model.pth'")