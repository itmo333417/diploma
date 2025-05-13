import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def make_unique_columns(columns):
    seen = {}
    unique_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_columns.append(col)
    return unique_columns


def augment_data(df, num_variations=5):
    non_score_cols = ['Номер', 'Итого', 'Пусто', 'ФИО', 'Группа', 'Поток']
    score_columns = [col for col in df.columns if col not in non_score_cols]

    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(num_variations):
            new_row = row.copy()
            num_missing = np.random.randint(1, max(2, len(score_columns) // 3))
            cols_to_null = np.random.choice(score_columns, size=num_missing, replace=False)

            if 'Экзамен' in score_columns and np.random.rand() < 0.6:
                new_row['Экзамен'] = np.nan

            for col in cols_to_null:
                new_row[col] = np.nan

            augmented_rows.append(new_row)
    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8", sep=None, engine="python")
    df.columns = make_unique_columns(df.columns)
    print("Столбцы после очистки (обучающие данные):", df.columns.tolist())

    df = augment_data(df)

    non_score_cols = ['Номер', 'Итого', 'Пусто', 'ФИО', 'Группа', 'Поток']
    score_columns = [col for col in df.columns if col not in non_score_cols]

    df[score_columns] = df[score_columns].apply(pd.to_numeric, errors='coerce')

    df['Текущая сумма'] = df[score_columns].sum(axis=1, skipna=True)

    max_scores = {
        'Лабораторная работа №1': 10,
        'Лабораторная работа №2': 10,
        'Лабораторная работа №3': 10,
        'Лабораторная работа №4': 10,
        'Лабораторная работа №5': 10,
        'Лабораторная работа №6': 10,
        'Промежуточное тестирование 1': 10,
        'Промежуточное тестирование 2': 10,
        'Экзамен': 20,
        'Дополнительные баллы': 3
    }

    df['Потерянные баллы'] = 0
    for col in score_columns:
        if col in max_scores:
            df['Потерянные баллы'] += np.where(df[col].notna(), max_scores[col] - df[col], 0)

    df = df.reset_index(drop=True)
    return df, score_columns, max_scores


def load_input_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8", sep=None, engine="python")
    df.columns = make_unique_columns(df.columns)
    return df


def train_model(training_file):
    df_train, score_columns, max_scores = load_and_clean_data(training_file)
    df_train = df_train.dropna(subset=['Итого'])

    non_feature_cols = ['Номер', 'Итого', 'Пусто']
    for col in ['ФИО', 'Группа', 'Поток']:
        if col in df_train.columns:
            non_feature_cols.append(col)

    X = df_train.drop(columns=non_feature_cols, errors='ignore')
    y = df_train['Итого']

    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\U0001F4CA MSE: {mse:.2f}")
    print(f"\U0001F4C9 MAE: {mae:.2f}")
    print(f"\U0001F4C8 R²: {r2:.4f}")

    joblib.dump(model, "trained_model.pkl")
    with open("feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    return model, feature_names, max_scores


def predict_row(row, model, feature_names, max_scores):
    row = row.copy()
    ignore_cols = {"Текущая аттестация"}

    # Приводим все входные значения к числовому типу
    input_scores = {
        feature: pd.to_numeric(row.get(feature), errors='coerce')
        for feature in feature_names
    }

    # Вычисляем полный балл по всем компонентам (только для тех, что заданы в max_scores)
    total_possible = sum(max_scores[k] for k in max_scores if k not in ignore_cols)

    # Вычисляем потерянные баллы по сданным компонентам
    lost_points = sum(
        max_scores[k] - input_scores[k]
        for k in max_scores
        if k not in ignore_cols and not pd.isna(input_scores.get(k))
    )

    # Максимальный итоговый балл с учётом потерь
    max_possible_score = total_possible - lost_points

    # Сумма уже набранных баллов (только по тем полям, которые не игнорируются)
    current_sum = sum(
        input_scores[k]
        for k in input_scores
        if k not in ignore_cols and not pd.isna(input_scores[k])
    )

    if all(not pd.isna(v) for v in input_scores.values()):
        predicted_score = current_sum
    else:
        # Прогнозируем недостающие баллы
        input_df = pd.DataFrame([input_scores])[feature_names]
        missing_score = model.predict(input_df)[0]
        # Итоговый результат не может быть ниже текущей суммы или выше max_possible_score
        predicted_score = max(current_sum, min(current_sum + missing_score, max_possible_score))

    # Нормализуем предсказанную оценку так, чтобы максимум был равен 103
    predicted_score = predicted_score * (103 / total_possible)

    return round(predicted_score,1)


def predict_scores_for_file(input_df, model, feature_names, max_scores):
    predicted_scores = input_df.apply(lambda row: predict_row(row, model, feature_names, max_scores), axis=1)
    input_df["Предсказанные баллы"] = predicted_scores
    input_df["Зона риска"] = input_df["Предсказанные баллы"].apply(lambda score: "Y" if score < 60 else "N")


    return input_df



if __name__ == "__main__":
    training_file = "data_train.csv"
    print("Обучение модели на данных из:", training_file)
    model, feature_names, max_scores = train_model(training_file)

    input_file = "data_test_journal.csv"
    print("Загружаем входной файл для предсказания:", input_file)
    df_input = load_input_data(input_file)

    df_output = predict_scores_for_file(df_input, model, feature_names, max_scores)

    print("\nДанные теста – Итоги предсказания:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df_output)

    output_file = os.path.splitext(input_file)[0] + "_predicted.csv"
    df_output.to_csv(output_file, index=False, encoding="utf-8")
    print("\nФайл с предсказанными баллами сохранён как:", output_file)