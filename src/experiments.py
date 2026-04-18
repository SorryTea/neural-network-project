"""
Plik odpowiada za uruchamianie eksperymentów i porównywanie modeli.
"""

import numpy as np

from src.models import KNNRegressor, NeuralNetworkRegressor
from src.metrics import calculate_all_metrics
from src.data_preparation import prepare_data, inverse_min_max_scale_target


DEFAULT_VALIDATION_SHARE = 0.15 / 0.85

DEFAULT_NN_CONFIG = {
    "hidden_layers": [64],
    "activation": "relu",
    "learning_rate": 0.01,
    "epochs": 2000,
    "initialization": "small_random",
    "batch_size": "full_batch",
}

DEFAULT_STUDY_SEEDS = [42, 123, 2024]

FINAL_NN_CANDIDATES = [
    {
        "hidden_layers": [64],
        "activation": "relu",
        "learning_rate": 0.01,
        "epochs": 2000,
        "initialization": "small_random",
        "batch_size": "full_batch",
    },
    {
        "hidden_layers": [64],
        "activation": "relu",
        "learning_rate": 0.05,
        "epochs": 2000,
        "initialization": "small_random",
        "batch_size": "full_batch",
    },
    {
        "hidden_layers": [64],
        "activation": "relu",
        "learning_rate": 0.01,
        "epochs": 4000,
        "initialization": "small_random",
        "batch_size": "full_batch",
    },
]


def describe_hidden_layers(hidden_layers):
    return "[" + ", ".join(str(size) for size in hidden_layers) + "]"


def get_primary_hidden_size(hidden_layers):
    return hidden_layers[0]


def normalize_batch_size(batch_size):
    if batch_size in [None, "full_batch"]:
        return "full_batch"
    return int(batch_size)


def resolve_hidden_layers(hidden_size=None, hidden_layers=None):
    if hidden_layers is not None:
        return list(hidden_layers)
    if isinstance(hidden_size, (list, tuple)):
        return list(hidden_size)
    return [hidden_size]


def build_nn_result_row(config, evaluation):
    hidden_layers = resolve_hidden_layers(
        hidden_size=config.get("hidden_size", 64),
        hidden_layers=config.get("hidden_layers"),
    )

    validation_metrics = evaluation["validation_metrics"]
    test_metrics = evaluation["test_metrics"]
    train_metrics = evaluation["train_metrics"]

    return {
        "hidden_layers": describe_hidden_layers(hidden_layers),
        "hidden_size": get_primary_hidden_size(hidden_layers),
        "activation": config["activation"],
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "initialization": config.get("initialization", "small_random"),
        "batch_size": normalize_batch_size(config.get("batch_size", "full_batch")),
        "TRAIN_MAPE": train_metrics["MAPE"],
        "VAL_MAPE": validation_metrics["MAPE"] if validation_metrics is not None else None,
        "TEST_MAPE": test_metrics["MAPE"],
        "RMSE": test_metrics["RMSE"],
        "MAE": test_metrics["MAE"],
        "MAPE": test_metrics["MAPE"],
    }


def evaluate_knn_configuration(X_train, y_train, X_test, y_test, k, distance_metric, weighted):
    model = KNNRegressor(k=k, distance_metric=distance_metric, weighted=weighted)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = calculate_all_metrics(y_train, y_pred_train)
    test_metrics = calculate_all_metrics(y_test, y_pred_test)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def run_knn_experiments(X_train, y_train, X_test, y_test):
    results = []

    k_values = [1, 3, 5, 9]
    distance_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    weighted_options = [False, True]

    for k in k_values:
        for distance_metric in distance_metrics:
            for weighted in weighted_options:
                evaluation = evaluate_knn_configuration(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    k=k,
                    distance_metric=distance_metric,
                    weighted=weighted,
                )

                train_metrics = evaluation["train_metrics"]
                test_metrics = evaluation["test_metrics"]

                results.append(
                    {
                        "model": "KNN",
                        "k": k,
                        "distance_metric": distance_metric,
                        "weighted": weighted,
                        "TRAIN_MSE": train_metrics["MSE"],
                        "TRAIN_RMSE": train_metrics["RMSE"],
                        "TRAIN_MAE": train_metrics["MAE"],
                        "TRAIN_MAPE": train_metrics["MAPE"],
                        "TEST_MSE": test_metrics["MSE"],
                        "TEST_RMSE": test_metrics["RMSE"],
                        "TEST_MAE": test_metrics["MAE"],
                        "TEST_MAPE": test_metrics["MAPE"],
                        "MSE": test_metrics["MSE"],
                        "RMSE": test_metrics["RMSE"],
                        "MAE": test_metrics["MAE"],
                        "MAPE": test_metrics["MAPE"],
                    }
                )

    return results


def evaluate_neural_network_configuration(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
    hidden_size=64,
    hidden_layers=None,
    activation="relu",
    learning_rate=0.01,
    epochs=2000,
    initialization="small_random",
    batch_size="full_batch",
    random_seed=42,
    early_stopping=True,
):
    """
    Trenuje sieć neuronową na zeskalowanym y_train.
    Jeśli podano walidację, używa jej do prostego early stopping.
    Wszystkie raportowane metryki są liczone po odwróceniu skali targetu.
    """
    resolved_hidden_layers = resolve_hidden_layers(hidden_size, hidden_layers)

    model = NeuralNetworkRegressor(
        input_size=X_train.shape[1],
        hidden_size=resolved_hidden_layers[0],
        hidden_layers=resolved_hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        activation=activation,
        initialization=initialization,
        batch_size=batch_size,
        random_seed=random_seed,
        early_stopping_patience=200,
    )

    model.fit(
        X_train,
        y_train_scaled,
        X_val=X_val,
        y_val=y_val_scaled,
        verbose=False,
        early_stopping=early_stopping,
    )

    y_pred_train_scaled = model.predict(X_train)
    y_pred_val_scaled = model.predict(X_val)
    y_pred_test_scaled = model.predict(X_test)

    y_pred_train = inverse_min_max_scale_target(
        y_pred_train_scaled, target_scaling_params
    )
    y_pred_val = inverse_min_max_scale_target(y_pred_val_scaled, target_scaling_params)
    y_pred_test = inverse_min_max_scale_target(
        y_pred_test_scaled, target_scaling_params
    )

    train_metrics = calculate_all_metrics(y_train, y_pred_train)
    validation_metrics = calculate_all_metrics(y_val, y_pred_val)
    test_metrics = calculate_all_metrics(y_test, y_pred_test)

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "y_pred_train": y_pred_train,
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": model,
    }


def run_neural_network_experiments(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    """
    Uruchamia siatkę eksperymentów dla podstawowej rodziny sieci neuronowych.
    Najlepszy model powinien być wybierany po VAL_MAPE.
    """
    results = []

    hidden_layer_options = [[8], [16], [32], [64]]
    activations = ["relu", "tanh"]
    learning_rates = [0.01, 0.001]
    epochs_values = [1000, 2000, 3000, 4000]

    for hidden_layers in hidden_layer_options:
        for activation in activations:
            for learning_rate in learning_rates:
                for epochs in epochs_values:
                    config = {
                        "hidden_layers": hidden_layers,
                        "activation": activation,
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "initialization": "small_random",
                        "batch_size": "full_batch",
                    }

                    evaluation = evaluate_neural_network_configuration(
                        X_train=X_train,
                        y_train=y_train,
                        y_train_scaled=y_train_scaled,
                        X_val=X_val,
                        y_val=y_val,
                        y_val_scaled=y_val_scaled,
                        X_test=X_test,
                        y_test=y_test,
                        target_scaling_params=target_scaling_params,
                        hidden_layers=config["hidden_layers"],
                        activation=config["activation"],
                        learning_rate=config["learning_rate"],
                        epochs=config["epochs"],
                        initialization=config["initialization"],
                        batch_size=config["batch_size"],
                        random_seed=42,
                        early_stopping=True,
                    )

                    results.append(build_nn_result_row(config, evaluation))

    return results


def run_final_neural_network_candidates(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
    candidate_configs=None,
    random_seed=42,
):
    """
    Uruchamia końcowe porównanie wybranych kandydatów sieci neuronowej.
    """
    if candidate_configs is None:
        candidate_configs = FINAL_NN_CANDIDATES

    results = []

    for config in candidate_configs:
        evaluation = evaluate_neural_network_configuration(
            X_train=X_train,
            y_train=y_train,
            y_train_scaled=y_train_scaled,
            X_val=X_val,
            y_val=y_val,
            y_val_scaled=y_val_scaled,
            X_test=X_test,
            y_test=y_test,
            target_scaling_params=target_scaling_params,
            hidden_layers=config["hidden_layers"],
            activation=config["activation"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            initialization=config["initialization"],
            batch_size=config["batch_size"],
            random_seed=random_seed,
            early_stopping=True,
        )

        validation_metrics = evaluation["validation_metrics"]
        train_metrics = evaluation["train_metrics"]
        test_metrics = evaluation["test_metrics"]
        hidden_layers = config["hidden_layers"]

        results.append(
            {
                "hidden_size": get_primary_hidden_size(hidden_layers),
                "activation": config["activation"],
                "learning_rate": config["learning_rate"],
                "epochs": config["epochs"],
                "TRAIN_MAPE": train_metrics["MAPE"],
                "TEST_MAPE": test_metrics["MAPE"],
                "VAL_MAPE": validation_metrics["MAPE"],
                "RMSE": test_metrics["RMSE"],
                "MAE": test_metrics["MAE"],
            }
        )

    return results


def _aggregate_metric_values(values):
    values_array = np.array(values, dtype=float)

    return {
        "Mean": float(np.mean(values_array)),
        "Std": float(np.std(values_array)),
        "Min": float(np.min(values_array)),
        "Max": float(np.max(values_array)),
    }


def format_tested_value(parameter_name, tested_value):
    if parameter_name == "hidden_layers":
        return describe_hidden_layers(tested_value)
    if parameter_name == "batch_size":
        return normalize_batch_size(tested_value)
    return tested_value


def run_neural_network_parameter_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
    parameter_name,
    tested_values,
    tested_value_labels=None,
    base_config=None,
    random_seeds=None,
):
    """
    Uruchamia analizę jednego parametru sieci neuronowej.
    Każda konfiguracja jest powtarzana 3 razy dla różnych seedów.
    W tabeli raportowane jest MAPE na TRAIN i TEST.
    """
    if base_config is None:
        base_config = DEFAULT_NN_CONFIG.copy()
    else:
        base_config = base_config.copy()

    if random_seeds is None:
        random_seeds = DEFAULT_STUDY_SEEDS

    results = []

    for index, tested_value in enumerate(tested_values):
        config = base_config.copy()
        config[parameter_name] = tested_value

        train_mape_values = []
        test_mape_values = []

        for seed in random_seeds:
            evaluation = evaluate_neural_network_configuration(
                X_train=X_train,
                y_train=y_train,
                y_train_scaled=y_train_scaled,
                X_val=X_val,
                y_val=y_val,
                y_val_scaled=y_val_scaled,
                X_test=X_test,
                y_test=y_test,
                target_scaling_params=target_scaling_params,
                hidden_layers=resolve_hidden_layers(
                    hidden_layers=config.get("hidden_layers"),
                    hidden_size=config.get("hidden_size", 64),
                ),
                activation=config["activation"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                initialization=config.get("initialization", "small_random"),
                batch_size=config.get("batch_size", "full_batch"),
                random_seed=seed,
                early_stopping=True,
            )

            train_mape_values.append(evaluation["train_metrics"]["MAPE"])
            test_mape_values.append(evaluation["test_metrics"]["MAPE"])

        train_stats = _aggregate_metric_values(train_mape_values)
        test_stats = _aggregate_metric_values(test_mape_values)

        if tested_value_labels is None:
            tested_value_label = format_tested_value(parameter_name, tested_value)
        else:
            tested_value_label = tested_value_labels[index]

        results.append(
            {
                "tested_value": tested_value_label,
                "Train Mean": train_stats["Mean"],
                "Train Std": train_stats["Std"],
                "Train Min": train_stats["Min"],
                "Train Max": train_stats["Max"],
                "Test Mean": test_stats["Mean"],
                "Test Std": test_stats["Std"],
                "Test Min": test_stats["Min"],
                "Test Max": test_stats["Max"],
            }
        )

    return results


def run_learning_rate_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="learning_rate",
        tested_values=[0.001, 0.005, 0.01, 0.05],
    )


def run_hidden_size_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="hidden_layers",
        tested_values=[[8], [16], [32], [64]],
        tested_value_labels=[8, 16, 32, 64],
    )


def run_activation_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="activation",
        tested_values=["sigmoid", "relu", "tanh", "leaky_relu"],
    )


def run_epochs_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="epochs",
        tested_values=[1000, 2000, 3000, 4000],
    )


def run_hidden_layers_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    tested_values = [
        [64],
        [64, 32],
        [64, 32, 16],
        [64, 32, 16, 8],
    ]
    tested_value_labels = [
        "1 hidden layer",
        "2 hidden layers",
        "3 hidden layers",
        "4 hidden layers",
    ]

    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="hidden_layers",
        tested_values=tested_values,
        tested_value_labels=tested_value_labels,
    )


def run_initialization_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="initialization",
        tested_values=["zeros", "small_random", "xavier", "he"],
    )


def run_batch_size_study(
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return run_neural_network_parameter_study(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        parameter_name="batch_size",
        tested_values=["full_batch", 32, 64, 128],
    )


def run_split_ratio_study(file_path, lag_count=3, base_config=None, random_seeds=None):
    """
    Analiza wpływu chronologicznego splitu train+val / test.
    Walidacja jest dalej wydzielana z części treningowej.
    """
    if base_config is None:
        base_config = DEFAULT_NN_CONFIG.copy()
    else:
        base_config = base_config.copy()

    if random_seeds is None:
        random_seeds = DEFAULT_STUDY_SEEDS

    tested_ratios = [0.5, 0.6, 0.7, 0.8]
    tested_labels = ["50/50", "60/40", "70/30", "80/20"]

    results = []

    for index, non_test_ratio in enumerate(tested_ratios):
        validation_ratio = non_test_ratio * DEFAULT_VALIDATION_SHARE
        train_ratio = non_test_ratio - validation_ratio

        prepared_data = prepare_data(
            file_path=file_path,
            lag_count=lag_count,
            train_ratio=train_ratio,
            val_ratio=validation_ratio,
        )

        train_mape_values = []
        test_mape_values = []

        for seed in random_seeds:
            evaluation = evaluate_neural_network_configuration(
                X_train=prepared_data["X_train_scaled"],
                y_train=prepared_data["y_train"],
                y_train_scaled=prepared_data["y_train_scaled"],
                X_val=prepared_data["X_val_scaled"],
                y_val=prepared_data["y_val"],
                y_val_scaled=prepared_data["y_val_scaled"],
                X_test=prepared_data["X_test_scaled"],
                y_test=prepared_data["y_test"],
                target_scaling_params=prepared_data["target_scaling_params"],
                hidden_layers=resolve_hidden_layers(
                    hidden_layers=base_config.get("hidden_layers"),
                    hidden_size=base_config.get("hidden_size", 64),
                ),
                activation=base_config["activation"],
                learning_rate=base_config["learning_rate"],
                epochs=base_config["epochs"],
                initialization=base_config.get("initialization", "small_random"),
                batch_size=base_config.get("batch_size", "full_batch"),
                random_seed=seed,
                early_stopping=True,
            )

            train_mape_values.append(evaluation["train_metrics"]["MAPE"])
            test_mape_values.append(evaluation["test_metrics"]["MAPE"])

        train_stats = _aggregate_metric_values(train_mape_values)
        test_stats = _aggregate_metric_values(test_mape_values)

        results.append(
            {
                "tested_value": tested_labels[index],
                "Train Mean": train_stats["Mean"],
                "Train Std": train_stats["Std"],
                "Train Min": train_stats["Min"],
                "Train Max": train_stats["Max"],
                "Test Mean": test_stats["Mean"],
                "Test Std": test_stats["Std"],
                "Test Min": test_stats["Min"],
                "Test Max": test_stats["Max"],
            }
        )

    return results


def run_all_neural_network_parameter_studies(
    file_path,
    X_train,
    y_train,
    y_train_scaled,
    X_val,
    y_val,
    y_val_scaled,
    X_test,
    y_test,
    target_scaling_params,
):
    return {
        "learning_rate": run_learning_rate_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
        "hidden_size": run_hidden_size_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
        "activation": run_activation_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
        "epochs": run_epochs_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
        "hidden_layers": run_hidden_layers_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
        "split_ratio": run_split_ratio_study(file_path=file_path),
        "initialization": run_initialization_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
        "batch_size": run_batch_size_study(
            X_train,
            y_train,
            y_train_scaled,
            X_val,
            y_val,
            y_val_scaled,
            X_test,
            y_test,
            target_scaling_params,
        ),
    }


def run_random_forest_parameter_study(X_train, y_train, X_test, y_test, parameter_name, tested_values):
    from sklearn.ensemble import RandomForestRegressor

    base_config = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    results = []

    for value in tested_values:
        config = base_config.copy()
        config[parameter_name] = value

        model = RandomForestRegressor(**config)
        model.fit(X_train, y_train)

        train_metrics = calculate_all_metrics(y_train, model.predict(X_train))
        test_metrics = calculate_all_metrics(y_test, model.predict(X_test))

        results.append({
            "tested_value": value,
            "TRAIN_MSE": train_metrics["MSE"],
            "TRAIN_RMSE": train_metrics["RMSE"],
            "TRAIN_MAE": train_metrics["MAE"],
            "TRAIN_MAPE": train_metrics["MAPE"],
            "TEST_MSE": test_metrics["MSE"],
            "TEST_RMSE": test_metrics["RMSE"],
            "TEST_MAE": test_metrics["MAE"],
            "TEST_MAPE": test_metrics["MAPE"],
        })

    return results


def run_rf_n_estimators_study(X_train, y_train, X_test, y_test):
    return run_random_forest_parameter_study(
        X_train, y_train, X_test, y_test,
        parameter_name="n_estimators",
        tested_values=[10, 50, 100, 200],
    )


def run_rf_max_depth_study(X_train, y_train, X_test, y_test):
    return run_random_forest_parameter_study(
        X_train, y_train, X_test, y_test,
        parameter_name="max_depth",
        tested_values=[2, 5, 10, 20],
    )


def get_best_result(results, metric_name="MAPE"):
    return min(results, key=lambda x: x[metric_name])
