import re
import pandas as pd


# Входной и выходной файлы
INPUT_CSV = "output/documentation_dataset_labeled.csv"
OUTPUT_CSV = "output/documentation_dataset_labeled_chunks.csv"


# Текст баннера, повторяющийся в начале страниц (может иметь разные пробелы/переносы)
BANNER_TEXT = (
    "Была ли статья полезной? Да Нет Спасибо за вашу оценку! Оставляя более подробный отзыв, вы помогаете нам"
    " улучшать документацию Комментарий к оценке Контактная информация (необязательно) Вложения Добавить файл"
    " Отменить Отправить"
)


def build_banner_regex(text: str) -> re.Pattern:
    """Строит regex, допускающий произвольные пробелы/переносы между словами баннера.

    Сопоставляет баннер только в начале строки.
    """
    words = re.findall(r"\S+", text)
    # Разрешаем любые пробелы/переносы между словами, игнорируем регистр
    pattern = r"^\s*" + r"\s+".join(map(re.escape, words)) + r"\s*"
    return re.compile(pattern, flags=re.IGNORECASE)


BANNER_REGEX = build_banner_regex(BANNER_TEXT)


def remove_banner_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Удаляем только один раз в начале
    return BANNER_REGEX.sub("", text, count=1)


def normalize_spaces(text: str) -> str:
    # Схлопываем множественные пробелы и невидимые пробелы
    return re.sub(r"\s+", " ", str(text)).strip()


def split_into_sentences(text: str) -> list:
    """Грубое разбиение на предложения по пунктуации . ! ? … с учётом пробелов.
    Если пунктуации нет, возвращает исходную строку как одно "предложение".
    """
    if not text:
        return []
    # Сначала нормализуем переносы
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[\.!\?…])\s+", cleaned)
    # Дополнительно делим по точкам с запятой и двоеточиям только если очень длинные куски
    refined = []
    for part in parts:
        if len(part) > 600:
            refined.extend(re.split(r"(?<=[;:])\s+", part))
        else:
            refined.append(part)
    # Фильтруем короткие/пустые фрагменты
    sentences = [s.strip() for s in refined if s and len(s.strip()) >= 2]
    return sentences


def main():
    # Читаем исходный датасет
    df = pd.read_csv(INPUT_CSV)

    # Берём только необходимые столбцы, пропуская строки без контента
    if not {"content", "category_name"}.issubset(df.columns):
        raise ValueError("Входной CSV должен содержать столбцы 'content' и 'category_name'")

    data = df[["content", "category_name"]].copy()
    data.dropna(subset=["content"], inplace=True)

    # Очищаем баннер и разбиваем на предложения
    records = []
    for _, row in data.iterrows():
        content = str(row["content"]) if pd.notna(row["content"]) else ""
        category = row["category_name"]

        content_wo_banner = remove_banner_prefix(content)
        sentences = split_into_sentences(content_wo_banner)

        for sent in sentences:
            sent_norm = normalize_spaces(sent)
            if sent_norm:
                records.append({
                    "content": sent_norm,
                    "category_name": category,
                })

    result_df = pd.DataFrame.from_records(records, columns=["content", "category_name"])

    # Удаляем дубликаты строк на всякий случай
    result_df.drop_duplicates(inplace=True)

    # Сохраняем
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Готово. Сохранено {len(result_df)} строк в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
