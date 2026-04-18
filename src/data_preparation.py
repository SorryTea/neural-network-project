"""
Przygotowanie danych do problemu regresyjnego dla Bitcoina.

Zakres pliku:
- wczytanie danych z CSV,
- usunięcie niepotrzebnych kolumn,
- sortowanie po dacie,
- tworzenie cech wejściowych,
- tworzenie lagów dla ceny zamknięcia,
- tworzenie targetu jako ceny zamknięcia z kolejnego dnia,
- budowa macierzy X i wektora y,
- podział danych na train/validation/test,
- normalizacja danych wejściowych.

Efekt końcowy:
Plik zwraca gotowe dane do trenowania i testowania modeli regresyjnych.
"""

import csv
import numpy as np


def load_data(file_path):
    """
    Wczytuje dane z pliku CSV i zwraca listę słowników.
    """
    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data


def clean_data(data):
    """
    Czyści dane:
    - zostawia tylko potrzebne kolumny,
    - konwertuje wartości liczbowe na float,
    - sortuje dane rosnąco po dacie.
    """
    cleaned_data = []

    for row in data:
        cleaned_row = {
            "Date": row["Date"],
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
            "Marketcap": float(row["Marketcap"]),
        }
        cleaned_data.append(cleaned_row)

    # Sortowanie po dacie rosnąco
    cleaned_data.sort(key=lambda x: x["Date"])

    return cleaned_data


def add_lag_features(data, lag_count=3):
    """
    Dodaje lagi dla kolumny Close.
    Przykład dla lag_count=3:
    - Close_lag_1
    - Close_lag_2
    - Close_lag_3
    """
    for i in range(len(data)):
        for lag in range(1, lag_count + 1):
            if i - lag >= 0:
                data[i][f"Close_lag_{lag}"] = data[i - lag]["Close"]
            else:
                data[i][f"Close_lag_{lag}"] = None

    return data


def add_derived_features(data):
    """
    Dodaje dodatkowe cechy pomocnicze:
    - dzienny zakres ceny: High - Low
    - różnica Open - Close
    - procentowa zmiana ceny zamknięcia względem poprzedniego dnia
    - średnia z 3 poprzednich zamknięć
    - średnia z 7 poprzednich zamknięć
    """
    for i in range(len(data)):
        # Zakres dzienny
        data[i]["High_Low_diff"] = data[i]["High"] - data[i]["Low"]

        # Różnica otwarcia i zamknięcia
        data[i]["Open_Close_diff"] = data[i]["Open"] - data[i]["Close"]

        # Procentowa zmiana ceny zamknięcia względem poprzedniego dnia
        if i - 1 >= 0 and data[i - 1]["Close"] != 0:
            data[i]["Close_change"] = (data[i]["Close"] - data[i - 1]["Close"]) / data[
                i - 1
            ]["Close"]
        else:
            data[i]["Close_change"] = None

        # Średnia z 3 poprzednich zamknięć
        if i >= 3:
            closes_3 = [data[i - j]["Close"] for j in range(1, 4)]
            data[i]["Close_mean_3"] = sum(closes_3) / 3.0
        else:
            data[i]["Close_mean_3"] = None

        # Średnia z 7 poprzednich zamknięć
        if i >= 7:
            closes_7 = [data[i - j]["Close"] for j in range(1, 8)]
            data[i]["Close_mean_7"] = sum(closes_7) / 7.0
        else:
            data[i]["Close_mean_7"] = None

    return data


def create_target(data):
    """
    Tworzy target jako cenę zamknięcia z następnego dnia.
    Dla obserwacji z dnia t targetem jest Close z dnia t+1.
    """
    for i in range(len(data)):
        if i + 1 < len(data):
            data[i]["Target"] = data[i + 1]["Close"]
        else:
            data[i]["Target"] = None

    return data


def build_feature_matrix_and_target(data):
    """
    Buduje macierz cech X i wektor y.
    Usuwa obserwacje, dla których brakuje którejkolwiek wymaganej cechy albo targetu.
    """
    feature_names = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Marketcap",
        "Close_lag_1",
        "Close_lag_2",
        "Close_lag_3",
        "High_Low_diff",
        "Open_Close_diff",
        "Close_change",
        "Close_mean_3",
        "Close_mean_7",
    ]

    X = []
    y = []
    dates = []

    for row in data:
        row_values = []
        valid_row = True

        for feature in feature_names:
            value = row.get(feature, None)
            if value is None:
                valid_row = False
                break
            row_values.append(value)

        if valid_row and row["Target"] is not None:
            X.append(row_values)
            y.append(row["Target"])
            dates.append(row["Date"])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    return X, y, dates, feature_names


def split_data(X, y, dates, train_ratio=0.7, val_ratio=0.15):
    """
    Dzieli dane chronologicznie na train / validation / test.
    Bez losowego mieszania.
    """
    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    dates_train = dates[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    dates_val = dates[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]
    dates_test = dates[val_end:]

    return (
        X_train,
        y_train,
        dates_train,
        X_val,
        y_val,
        dates_val,
        X_test,
        y_test,
        dates_test,
    )


def min_max_scale(X_train, X_val, X_test):
    """
    Normalizacja min-max na podstawie zbioru treningowego.
    Każda cecha jest skalowana osobno do zakresu [0, 1].
    """
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)

    ranges = max_vals - min_vals

    # Zabezpieczenie przed dzieleniem przez zero
    for i in range(len(ranges)):
        if ranges[i] == 0:
            ranges[i] = 1.0

    X_train_scaled = (X_train - min_vals) / ranges
    X_val_scaled = (X_val - min_vals) / ranges
    X_test_scaled = (X_test - min_vals) / ranges

    scaling_params = {"min": min_vals, "max": max_vals, "range": ranges}

    return X_train_scaled, X_val_scaled, X_test_scaled, scaling_params


def min_max_scale_target(y_train, y_val, y_test):
    """
    Normalizacja min-max targetu na podstawie zbioru treningowego.
    Wszystkie wartości są skalowane do zakresu [0, 1].
    """
    y_train = np.array(y_train, dtype=float)
    y_val = np.array(y_val, dtype=float)
    y_test = np.array(y_test, dtype=float)

    min_val = np.min(y_train)
    max_val = np.max(y_train)
    range_val = max_val - min_val

    if range_val == 0:
        range_val = 1.0

    y_train_scaled = (y_train - min_val) / range_val
    y_val_scaled = (y_val - min_val) / range_val
    y_test_scaled = (y_test - min_val) / range_val

    scaling_params = {"min": min_val, "max": max_val, "range": range_val}

    return y_train_scaled, y_val_scaled, y_test_scaled, scaling_params


def inverse_min_max_scale_target(y_scaled, scaling_params):
    """
    Odwraca skalowanie min-max targetu do oryginalnej skali.
    """
    y_scaled = np.array(y_scaled, dtype=float)
    return y_scaled * scaling_params["range"] + scaling_params["min"]


def prepare_data(file_path, lag_count=3, train_ratio=0.7, val_ratio=0.15):
    """
    Główna funkcja przygotowująca dane.
    Wykonuje cały pipeline:
    1. Wczytanie danych
    2. Czyszczenie
    3. Lagi
    4. Dodatkowe cechy
    5. Target
    6. Budowa X i y
    7. Podział na train/val/test
    8. Normalizacja

    Zwraca słownik z gotowymi danymi.
    """
    data = load_data(file_path)
    data = clean_data(data)
    data = add_lag_features(data, lag_count=lag_count)
    data = add_derived_features(data)
    data = create_target(data)

    X, y, dates, feature_names = build_feature_matrix_and_target(data)

    (
        X_train,
        y_train,
        dates_train,
        X_val,
        y_val,
        dates_val,
        X_test,
        y_test,
        dates_test,
    ) = split_data(X, y, dates, train_ratio=train_ratio, val_ratio=val_ratio)

    X_train_scaled, X_val_scaled, X_test_scaled, scaling_params = min_max_scale(
        X_train, X_val, X_test
    )
    (
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        target_scaling_params,
    ) = min_max_scale_target(y_train, y_val, y_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "dates_train": dates_train,
        "X_val": X_val,
        "y_val": y_val,
        "dates_val": dates_val,
        "X_test": X_test,
        "y_test": y_test,
        "dates_test": dates_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "feature_names": feature_names,
        "scaling_params": scaling_params,
        "target_scaling_params": target_scaling_params,
    }


if __name__ == "__main__":
    file_path = "data/coin_Bitcoin.csv"
    prepared_data = prepare_data(file_path)

    print("Przygotowanie danych zakończone.")
    print(f"Liczba cech: {len(prepared_data['feature_names'])}")
    print(f"Cechy: {prepared_data['feature_names']}")
    print(f"Train: {prepared_data['X_train'].shape}, {prepared_data['y_train'].shape}")
    print(f"Validation: {prepared_data['X_val'].shape}, {prepared_data['y_val'].shape}")
    print(f"Test: {prepared_data['X_test'].shape}, {prepared_data['y_test'].shape}")
