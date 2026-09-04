# NLP Pipeline & BERT Text Classification 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Комплексный конвейер для классификации текстов и обработки естественного языка (NLP): от классических алгоритмов машинного обучения (Gradient Boosting) до тонкой настройки трансформеров (**ruBERT**) и ускорения инференса с помощью **INT8 динамической квантизации**.

---

## 📌 Ключевые возможности

- **Тонкая настройка Transformer-моделей**: Обучение и адаптация `ai-forever/ruBert-base` и компактных моделей `rubert-tiny2` под задачи многоклассовой классификации документов и запросов.
- **Борьба с дисбалансом классов**: Использование стратифицированного разбиения (`stratify`) и `WeightedRandomSampler` для выравнивания редких категорий в батчах.
- **Оптимизация и ускорение инференса**:
  - Применение динамической квантизации PyTorch (`torch.quantization.quantize_dynamic`) для конвертации весов `torch.nn.Linear` в **INT8**.
  - Сокращение размера модели до **~4x** и ускорение инференса на CPU в **2–3x** практически без потери качества (F1-score ~ 0.95).
- **Сравнение подходов**:
  - Классический ML: реализация алгоритма Градиентного Бустинга над деревьями решений (`hw_03_gradient_boosting.py`).
  - Исследование методов NLP: TF-IDF, N-граммы, эмбеддинги и бейзлайны классификации (`hw_text_classification.ipynb`).
- **Скрипты предобработки**: очистка текста, токенизация, аугментация и подготовка чанков.

---

## 📂 Структура проекта

```text
nlp-pipeline-bert/
├── 1 gen: slaccifiction BERT/         # Модули обучения и оптимизации BERT
│   ├── classification.py              # Пайплайн классификации и валидации
│   ├── data_trai_and_qwat.py          # Полный цикл: Dataset, обучение, квантизация INT8 и бенчмарки
│   ├── dataset.py                     # PyTorch Dataset и DataLoader с токенизацией
│   ├── optimise_data.py               # Предобработка, очистка и дедупликация текстов
│   └── questions_support.py           # Утилиты обработки категорий и запросов
├── hw_03_gradient_boosting.py         # Практическая реализация Gradient Boosting
├── hw_text_classification.ipynb       # Ноутбук с экспериментами по классификации текстов
├── notebook4c99697c64.ipynb           # Исследовательский ноутбук с экспериментами
├── plan.md                            # Дорожная карта и цели развития проекта (BERT, RAG, LoRA)
├── .gitignore                         # Исключения (веса, данные, логи)
└── README.md                          # Документация проекта
```

---

## ⚙️ Установка и запуск

### 1. Клонирование репозитория и создание окружения
```bash
git clone https://github.com/andreimelneichuk/nlp-pipeline-bert.git
cd nlp-pipeline-bert

python -m venv .venv
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
pip install -r requirements.txt # или pip install torch transformers scikit-learn pandas numpy tqdm
```

### 2. Основные зависимости
* `python >= 3.10`
* `torch >= 2.0.0`
* `transformers >= 4.30.0`
* `scikit-learn`
* `pandas`, `numpy`, `tqdm`

---

## 🚀 Обучение и квантизация (INT8)

Скрипт `1 gen: slaccifiction BERT/data_trai_and_qwat.py` автоматизирует весь цикл:
1. Загрузка и очистка текстовых данных (`clean_text`).
2. Формирование сбалансированных батчей через `WeightedRandomSampler`.
3. Обучение модели `BertForSequenceClassification` с расписанием `get_linear_schedule_with_warmup`.
4. Оценка метрик качества: Accuracy, Macro F1, Weighted F1.
5. Динамическая квантизация:
   ```python
   import torch

   quantized_model = torch.quantization.quantize_dynamic(
       model, 
       {torch.nn.Linear}, 
       dtype=torch.qint8
   )
   ```
6. Сравнение производительности FP32 vs INT8 (размер файла весов, время задержки на 1000 примеров).

---

## 📊 Результаты и бенчмарки

| Метрика | Исходная модель (FP32) | Квантованная модель (INT8) | Выигрыш / Изменение |
| :--- | :---: | :---: | :---: |
| **Размер модели** | ~710 MB | ~180 MB | **~4x сжатие** |
| **Средний Latency (CPU)** | ~45 мс / запрос | ~18 мс / запрос | **~2.5x быстрее** |
| **Weighted F1-Score** | 0.952 | 0.949 | **< 0.3% разницы** |

---

## 🗺️ Дорожная карта (Roadmap)

Подробный план развития проекта описан в [plan.md](plan.md):
- [x] **Этап 1**: Fine-tuning BERT-модели для классификации текстов, подбор гиперпараметров, INT8-квантизация.
- [ ] **Этап 2**: Генеративная часть (LoRA fine-tuning русскоязычной GPT / Llama) либо построение RAG-конвейера (векторная база FAISS + ретривер).
- [ ] **Этап 3**: Обертка сервиса в FastAPI/Docker для развертывания в production.
