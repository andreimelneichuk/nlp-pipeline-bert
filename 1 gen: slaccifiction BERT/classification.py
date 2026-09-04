import os
import time
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Конфигурация ---
OUTPUT_CSV = "output/documentation_dataset_chunks_labeled.csv"

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Задавать через .env или переменную окружения
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct")
LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Категории для классификации
CATEGORIES = {
    1: "Введение и начало работы",
    2: "Архитектура и безопасность",
    3: "Интерфейс и навигация",
    4: "Функциональные модули",
    5: "Обновления и релизы"
}

categories_text = "\n".join([f"{k}. {v}" for k, v in CATEGORIES.items()])


def query_llm_category(text: str) -> tuple[str | None, float | None]:
    """Запрашивает у LLM (через OpenRouter) категорию для текста. Возвращает (label, confidence=1.0) или (None, None)."""
    if not LLM_API_KEY:
        print("OPENROUTER_API_KEY не задан. Установите переменную окружения OPENROUTER_API_KEY (см. https://openrouter.ai).")
        return None, None

    system_msg = (
        "Ты ассистент по классификации коротких русских предложений документации. "
        "Тебе дан фрагмент текста (одно-два предложения). Твоя задача — выбрать РОВНО одну категорию из списка. "
        "Отвечай только НАЗВАНИЕМ категории, без пояснений. Категории:\n\n" + categories_text
    )
    user_msg = f"Текст: {text[:800]}"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 50  # ограничиваем длину ответа
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Обязательно для OpenRouter
        "X-Title": "Doc Classifier Script"
    }

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        resp = session.post(LLM_API_URL, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            print("Пустой ответ от OpenRouter API")
            return None, None

        # Сопоставление с категориями
        label = None
        for _, cat in CATEGORIES.items():
            if content.lower() == cat.lower():
                label = cat
                break
        if label is None:
            for _, cat in CATEGORIES.items():
                if cat.lower() in content.lower():
                    label = cat
                    break
        if label is None:
            print(f"Не удалось сопоставить ответ '{content}' с категориями")
            return None, None

        return label, 1.0

    except requests.exceptions.Timeout as e:
        print(f"Таймаут OpenRouter API: {e}")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка OpenRouter API: {e}")
        # Выводим тело ошибки, если есть
        try:
            error_detail = resp.json()
            print(f"Детали ошибки: {error_detail}")
        except:
            pass
        return None, None


def classify_chunk(content):
    """Классификация одного фрагмента через OpenRouter."""
    return query_llm_category(str(content))


def main():
    if not os.path.exists(OUTPUT_CSV):
        raise ValueError(f"Выходной файл {OUTPUT_CSV} не существует. Сначала создайте его.")

    df = pd.read_csv(OUTPUT_CSV)
    if 'content' not in df.columns:
        raise ValueError("Ожидается столбец 'content' в выходном CSV")

    if 'predicted_category_name' not in df.columns:
        df['predicted_category_name'] = ''
    if 'predicted_score' not in df.columns:
        df['predicted_score'] = 0.0

    total = len(df)
    print(f"Всего строк в файле: {total}")

    rows_to_process = []
    for idx, row in df.iterrows():
        existing_label = str(row.get('predicted_category_name', '')).strip()
        existing_score = float(row.get('predicted_score', 0.0))

        need_processing = False
        reason = ""

        if existing_score <= 0.0:
            need_processing = True
            reason = f"score={existing_score}"
        elif not existing_label:
            need_processing = True
            reason = "пустая категория"
        elif existing_label not in CATEGORIES.values():
            need_processing = True
            reason = f"неизвестная категория: {existing_label}"

        if need_processing:
            rows_to_process.append((idx, row, reason))

    print(f"Строк для обработки: {len(rows_to_process)}")

    if not rows_to_process:
        print("Все строки уже обработаны корректно!")
        return

    for idx, row, reason in rows_to_process:
        row_id = int(row.get('row_id', idx))
        text = str(row['content'])

        print(f"Обрабатывается строка {idx+1}/{total} (row_id={row_id}) - {reason}")

        label, score = classify_chunk(text)

        if label is None:
            label = ""
            score = 0.0
            print(f"Не удалось получить предсказание для row_id={row_id}")
        else:
            print(f"Получено предсказание: {label} (score={score})")

        df.at[idx, 'predicted_category_name'] = label
        df.at[idx, 'predicted_score'] = score
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

        # Пауза между запросами (OpenRouter может ограничивать частоту)
        time.sleep(2)  # можно уменьшить до 1–2 сек, но не 0

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nГотово. Результаты сохранены в {OUTPUT_CSV}")

    try:
        processed_count = len(df[df['predicted_score'] > 0])
        empty_count = len(df[df['predicted_category_name'] == ''])
        print(f"\nСтатистика:")
        print(f"- Строк с хорошими предсказаниями: {processed_count}")
        print(f"- Строк с пустыми категориями: {empty_count}")

        if 'predicted_category_name' in df.columns:
            print("\nРаспределение по категориям:")
            print(df['predicted_category_name'].value_counts())
    except Exception as e:
        print(f"Ошибка при выводе статистики: {e}")


if __name__ == "__main__":
    main()