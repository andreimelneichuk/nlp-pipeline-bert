# NLP Pipeline & BERT Text Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)

Набор скриптов и ноутбуков по классификации текстов: от сбора и разметки датасета документации до тонкой настройки **ruBERT** и динамической **INT8-квантизации**. Отдельно лежат учебные эксперименты с RNN/GRU и градиентным бустингом (CatBoost).

---

## Что внутри

- **Сбор датасета**: краулер документации на Playwright + BeautifulSoup, сохраняет страницы в CSV (`bert_classification/dataset.py`).
- **Разметка через LLM**: классификация фрагментов текста по 5 категориям через OpenRouter API (`bert_classification/classification.py`).
- **Предобработка**: удаление повторяющегося баннера, разбиение на предложения, дедупликация (`bert_classification/optimise_data.py`).
- **Fine-tuning BERT** (`ai-forever/ruBert-base`): стратифицированное разбиение train/val/test, `WeightedRandomSampler` для редких классов, `AdamW` + `get_linear_schedule_with_warmup`, ранняя остановка по weighted F1.
- **Динамическая квантизация INT8** (`torch.quantization.quantize_dynamic` для `torch.nn.Linear`) и сравнение FP32 vs INT8: размер модели, время инференса на тестовой выборке, Accuracy и F1.
- **Учебные эксперименты**:
  - RNN/GRU-классификация новостей AG News (`rnn_gru_text_classification.ipynb`);
  - регрессия популярности треков: бейзлайн RandomForest и CatBoost с подбором глубины через кросс-валидацию (`catboost_popularity_regression.py`).

---

## Структура проекта

```text
nlp-pipeline-bert/
├── bert_classification/
│   ├── dataset.py                     # Краулер документации (Playwright) -> output/documentation_dataset.csv
│   ├── classification.py              # Разметка фрагментов по категориям через LLM (OpenRouter)
│   ├── optimise_data.py               # Очистка от баннера, разбиение на предложения, дедупликация
│   ├── questions_support.py           # Извлечение пар «вопрос — категория» из Excel-таблицы
│   └── train_and_quantize.py          # Dataset, обучение ruBERT, оценка, квантизация INT8, сравнение
├── bert_finetuning_quantization.ipynb # Тот же пайплайн обучения и квантизации в виде ноутбука
├── rnn_gru_text_classification.ipynb  # RNN/GRU на AG News (учебное задание)
├── catboost_popularity_regression.py  # RandomForest и CatBoost для регрессии (учебное задание)
├── plan.md                            # План работ по fine-tuning BERT
├── requirements.txt
└── .gitignore                         # Исключения (данные, веса, логи)
```

---

## Установка

```bash
git clone https://github.com/andreimelneichuk/nlp-pipeline-bert.git
cd nlp-pipeline-bert

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` покрывает обучение BERT. Для остальных скриптов нужны дополнительные пакеты:

- краулер: `pip install playwright beautifulsoup4 && playwright install chromium`;
- чтение Excel в `questions_support.py`: `openpyxl`;
- учебные эксперименты: `catboost`, `datasets`, `nltk`, `matplotlib`.

---

## Запуск

Датасеты и веса моделей в репозиторий не входят (см. `.gitignore`). Скрипты ожидают файлы в каталоге `output/` относительно текущей директории.

1. Сбор страниц документации. Адрес стартовой страницы задаётся в `BASE_URL`:
   ```bash
   python bert_classification/dataset.py
   ```
2. Разметка фрагментов через LLM. Нужен ключ OpenRouter:
   ```bash
   export OPENROUTER_API_KEY=...
   python bert_classification/classification.py
   ```
3. Очистка и разбиение на предложения:
   ```bash
   python bert_classification/optimise_data.py
   ```
4. Обучение, оценка и квантизация. Читает `output/documentation_dataset_labeled.csv` со столбцами `content` и `category_name`:
   ```bash
   python bert_classification/train_and_quantize.py
   ```

Скрипт обучения выполняет следующие шаги:
1. Загружает данные и делает лёгкую очистку текста (`clean_text`).
2. Формирует батчи через `WeightedRandomSampler`.
3. Обучает `BertForSequenceClassification` с `get_linear_schedule_with_warmup` и сохраняет лучшую модель в `best_model.pth`.
4. Считает Accuracy и weighted F1 на тестовой выборке.
5. Применяет динамическую квантизацию:
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```
6. Выводит размер модели, время инференса и метрики до и после квантизации и сохраняет `quantized_model.pth`.

Если в сборке PyTorch нет движка квантизации (например, на macOS), скрипт сообщает об этом и продолжает работу с исходной моделью.

---

## Результаты

В ноутбуке `rnn_gru_text_classification.ipynb` сохранены результаты на AG News (accuracy на 5000 примерах из тестовой части):

| Модель | Accuracy |
| :--- | :---: |
| RNN (бейзлайн) | 0.9002 |
| GRU | 0.9022 |
| GRU, 2 слоя | 0.9086 |
| GRU, 2 слоя, конкатенация среднего и последнего скрытого состояния | 0.9092 |
| то же + hidden_dim=256, dropout=0.5, 10 эпох | 0.9012 |

Метрики BERT и замеры FP32 vs INT8 скрипт `train_and_quantize.py` печатает при запуске. В репозитории они не сохранены.

---

## Дальнейшие шаги

План работ описан в [plan.md](plan.md). Экспорт в ONNX и упаковка инференса в Docker пока не реализованы.
