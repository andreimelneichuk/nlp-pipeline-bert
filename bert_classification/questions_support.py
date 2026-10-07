import pandas as pd

def extract_questions_and_categories(df: pd.DataFrame, sheet_name: str):
    """
    Функция анализирует таблицу и возвращает DataFrame с колонками
    content (вопрос) и category_name (категория), если удаётся определить.
    """
    df = df.dropna(how="all")  # удаляем пустые строки
    if df.empty:
        print(f"❌ Лист '{sheet_name}' — пустой, пропускаем")
        return pd.DataFrame(columns=["content", "category_name"])

    # Приводим названия столбцов к нижнему регистру для удобства
    cols_lower = [str(c).lower().strip() for c in df.columns]

    # Попытка найти колонки по названию
    question_cols = [col for col in df.columns if "вопрос" in str(col).lower() or "content" in str(col).lower()]
    category_cols = [col for col in df.columns if "категор" in str(col).lower() or "category" in str(col).lower()]

    # Если прямых совпадений нет — попробуем эвристически
    if not question_cols:
        # Часто вопрос бывает в первой или второй колонке с длинным текстом
        for col in df.columns:
            if df[col].astype(str).str.len().mean() > 20:  # длинные фразы
                question_cols.append(col)
                break

    if not category_cols:
        # Категория часто имеет ограниченное число уникальных значений
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique < 20:  # немного категорий
                category_cols.append(col)
                break

    if not question_cols or not category_cols:
        print(f"⚠️ Лист '{sheet_name}' — структура не определена (вопрос: {question_cols}, категория: {category_cols})")
        return pd.DataFrame(columns=["content", "category_name"])

    q_col = question_cols[0]
    c_col = category_cols[0]

    extracted = df[[q_col, c_col]].copy()
    extracted.columns = ["content", "category_name"]

    # Очищаем строки
    extracted = extracted.dropna(subset=["content", "category_name"])
    extracted["content"] = extracted["content"].astype(str).str.strip()
    extracted["category_name"] = extracted["category_name"].astype(str).str.strip()

    print(f"✅ Лист '{sheet_name}' — найдено {len(extracted)} записей (вопрос: '{q_col}', категория: '{c_col}')")
    return extracted


def main():
    file_path = 'data/support_scripts.xlsx'
    sheets = pd.read_excel(file_path, sheet_name=None)
    all_data = []

    for sheet_name, df in sheets.items():
        print(f"\n📄 Анализ листа: {sheet_name}")
        extracted = extract_questions_and_categories(df, sheet_name)
        if not extracted.empty:
            all_data.append(extracted)

    if not all_data:
        print("\n❌ Не удалось извлечь данные ни из одного листа.")
        return

    result = pd.concat(all_data, ignore_index=True)
    result.to_csv("questions_and_categories.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 Сохранено {len(result)} строк в 'questions_and_categories.csv'")


if __name__ == "__main__":
    main()