"""
Plik zawiera funkcje do oceny jakości modeli regresyjnych.

Metryki:
- MSE
- RMSE
- MAE
- MAPE
- funkcja zbiorcza zwracająca wszystkie metryki
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    Wynik w procentach.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    non_zero_mask = y_true != 0

    if np.sum(non_zero_mask) == 0:
        return 0.0

    y_true_non_zero = y_true[non_zero_mask]
    y_pred_non_zero = y_pred[non_zero_mask]

    return (
        np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100.0
    )


def calculate_all_metrics(y_true, y_pred):
    """
    Zwraca wszystkie metryki w postaci słownika.
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


if __name__ == "__main__":
    y_true = np.array([10, 20, 30, 40], dtype=float)
    y_pred = np.array([12, 18, 29, 41], dtype=float)

    results = calculate_all_metrics(y_true, y_pred)

    print("Test metryk:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")
