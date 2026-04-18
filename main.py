import sys

from src.data_preparation import prepare_data
from src.models import (
    BaselineLastValueRegressor,
    BaselineMovingAverageRegressor,
    LinearRegressionGD,
)
from src.metrics import calculate_all_metrics
from src.experiments import (
    run_knn_experiments,
    run_rf_n_estimators_study,
    run_rf_max_depth_study,
    run_neural_network_experiments,
    run_all_neural_network_parameter_studies,
    run_final_neural_network_candidates,
    get_best_result,
    evaluate_neural_network_configuration,
)
from src.utils import (
    print_section,
    print_table,
    ensure_directory,
    save_rows_to_csv,
    save_text_file,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def print_metrics(model_name, y_true, y_pred):
    """
    Wypisuje metryki dla danego modelu.
    """
    results = calculate_all_metrics(y_true, y_pred)

    print(f"\n{model_name}")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return results


def print_best_result(title, result):
    """
    Wypisuje najlepszy wynik eksperymentu w czytelnej formie.
    """
    print(f"\n{title}")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def build_method_result_row(method_name, train_metrics, test_metrics):
    return {
        "method": method_name,
        "TRAIN_MSE": train_metrics["MSE"],
        "TRAIN_RMSE": train_metrics["RMSE"],
        "TRAIN_MAE": train_metrics["MAE"],
        "TRAIN_MAPE": train_metrics["MAPE"],
        "TEST_MSE": test_metrics["MSE"],
        "TEST_RMSE": test_metrics["RMSE"],
        "TEST_MAE": test_metrics["MAE"],
        "TEST_MAPE": test_metrics["MAPE"],
    }


def build_summary_text(best_knn, best_nn, methods_ranking):
    lines = [
        "Bitcoin Regression Project - Summary",
        "",
        "Best k-NN configuration:",
        f"k: {best_knn['k']}",
        f"distance_metric: {best_knn['distance_metric']}",
        f"weighted: {best_knn['weighted']}",
        f"TRAIN_MAPE: {best_knn['TRAIN_MAPE']:.4f}",
        f"TEST_MAPE: {best_knn['TEST_MAPE']:.4f}",
        f"TEST_RMSE: {best_knn['TEST_RMSE']:.4f}",
        f"TEST_MAE: {best_knn['TEST_MAE']:.4f}",
        "",
        "Best neural network configuration (selected by validation MAPE):",
        f"hidden_layers: {best_nn['hidden_layers']}",
        f"activation: {best_nn['activation']}",
        f"learning_rate: {best_nn['learning_rate']:.4f}",
        f"epochs: {best_nn['epochs']}",
        f"initialization: {best_nn['initialization']}",
        f"batch_size: {best_nn['batch_size']}",
        f"TRAIN_MAPE: {best_nn['TRAIN_MAPE']:.4f}",
        f"VAL_MAPE: {best_nn['VAL_MAPE']:.4f}",
        f"TEST_MAPE: {best_nn['TEST_MAPE']:.4f}",
        f"RMSE: {best_nn['RMSE']:.4f}",
        f"MAE: {best_nn['MAE']:.4f}",
        "",
        "Final ranking of methods by TEST MAPE:",
    ]

    for index, row in enumerate(methods_ranking, start=1):
        lines.append(
            f"{index}. {row['method']} - TEST_MAPE: {row['TEST_MAPE']:.4f}, "
            f"TEST_RMSE: {row['TEST_RMSE']:.4f}, TEST_MAE: {row['TEST_MAE']:.4f}"
        )

    return "\n".join(lines)


def main():
    data_file_path = "data/coin_Bitcoin.csv"
    results_dir = "results"
    tables_dir = "results/tables"
    text_dir = "results/text"

    ensure_directory(results_dir)
    ensure_directory(tables_dir)
    ensure_directory(text_dir)

    # =========================
    # 1. Przygotowanie danych
    # =========================
    print_section("1. Przygotowanie danych")

    data = prepare_data(data_file_path)

    # Dane surowe - dla baseline
    X_train_raw = data["X_train"]
    X_test_raw = data["X_test"]

    # Dane znormalizowane - dla modeli uczonych
    X_train = data["X_train_scaled"]
    X_val = data["X_val_scaled"]
    X_test = data["X_test_scaled"]

    # Target w oryginalnej skali
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # Target w skali znormalizowanej - dla sieci neuronowej
    y_train_scaled = data["y_train_scaled"]
    y_val_scaled = data["y_val_scaled"]

    target_scaling_params = data["target_scaling_params"]

    print("Przygotowanie danych zakończone.")
    print(f"Liczba cech: {len(data['feature_names'])}")
    print(f"Nazwy cech: {data['feature_names']}")
    print()

    print("Rozmiary zbiorów:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    method_results = []

    # =========================
    # 2. Baseline: ostatnia wartość
    # =========================
    print_section("2. Baseline - ostatnia wartość")

    baseline_last = BaselineLastValueRegressor(close_lag_1_index=5)
    baseline_last.fit(X_train_raw, y_train)

    y_pred_train = baseline_last.predict(X_train_raw)
    y_pred_test = baseline_last.predict(X_test_raw)

    baseline_last_train_metrics = print_metrics(
        "Baseline Last Value - TRAIN", y_train, y_pred_train
    )
    baseline_last_test_metrics = print_metrics(
        "Baseline Last Value - TEST", y_test, y_pred_test
    )
    method_results.append(
        build_method_result_row(
            "Baseline Last Value",
            baseline_last_train_metrics,
            baseline_last_test_metrics,
        )
    )

    # =========================
    # 3. Baseline: średnia z lagów
    # =========================
    print_section("3. Baseline - średnia z lagów")

    baseline_avg = BaselineMovingAverageRegressor(lag_indices=[5, 6, 7])
    baseline_avg.fit(X_train_raw, y_train)

    y_pred_train = baseline_avg.predict(X_train_raw)
    y_pred_test = baseline_avg.predict(X_test_raw)

    baseline_avg_train_metrics = print_metrics(
        "Baseline Moving Average - TRAIN", y_train, y_pred_train
    )
    baseline_avg_test_metrics = print_metrics(
        "Baseline Moving Average - TEST", y_test, y_pred_test
    )
    method_results.append(
        build_method_result_row(
            "Baseline Moving Average",
            baseline_avg_train_metrics,
            baseline_avg_test_metrics,
        )
    )

    # =========================
    # 4. Regresja liniowa
    # =========================
    print_section("4. Regresja liniowa")

    linear_model = LinearRegressionGD(learning_rate=0.01, n_iterations=2000)
    linear_model.fit(X_train, y_train)

    y_pred_train = linear_model.predict(X_train)
    y_pred_test = linear_model.predict(X_test)

    linear_train_metrics = print_metrics("Linear Regression - TRAIN", y_train, y_pred_train)
    linear_test_metrics = print_metrics("Linear Regression - TEST", y_test, y_pred_test)
    method_results.append(
        build_method_result_row(
            "Linear Regression", linear_train_metrics, linear_test_metrics
        )
    )

    # =========================
    # 5. Eksperymenty k-NN i najlepszy wariant
    # =========================
    print_section("5. k-NN - eksperymenty i najlepszy wariant")

    knn_results = run_knn_experiments(X_train, y_train, X_test, y_test)
    knn_table_columns = [
        "model",
        "k",
        "distance_metric",
        "weighted",
        "TRAIN_MSE",
        "TRAIN_RMSE",
        "TRAIN_MAE",
        "TRAIN_MAPE",
        "TEST_MSE",
        "TEST_RMSE",
        "TEST_MAE",
        "TEST_MAPE",
    ]
    knn_results_sorted = sorted(knn_results, key=lambda result: result["TEST_MAPE"])
    print_table(
        "Wyniki eksperymentów k-NN",
        knn_results_sorted,
        columns=knn_table_columns,
    )

    best_knn = get_best_result(knn_results, metric_name="TEST_MAPE")
    print_best_result("Najlepszy wariant k-NN", best_knn)

    method_results.append(
        {
            "method": "KNN Best",
            "TRAIN_MSE": best_knn["TRAIN_MSE"],
            "TRAIN_RMSE": best_knn["TRAIN_RMSE"],
            "TRAIN_MAE": best_knn["TRAIN_MAE"],
            "TRAIN_MAPE": best_knn["TRAIN_MAPE"],
            "TEST_MSE": best_knn["TEST_MSE"],
            "TEST_RMSE": best_knn["TEST_RMSE"],
            "TEST_MAE": best_knn["TEST_MAE"],
            "TEST_MAPE": best_knn["TEST_MAPE"],
        }
    )

    # =========================
    # 6. Random Forest - analiza parametrów
    # =========================
    print_section("6. Random Forest - analiza parametrów")

    rf_param_columns = [
        "tested_value",
        "TRAIN_MSE", "TRAIN_RMSE", "TRAIN_MAE", "TRAIN_MAPE",
        "TEST_MSE", "TEST_RMSE", "TEST_MAE", "TEST_MAPE",
    ]

    rf_n_estimators_results = run_rf_n_estimators_study(X_train, y_train, X_test, y_test)
    print_table(
        "6.1 RF - liczba drzew (n_estimators)",
        rf_n_estimators_results,
        columns=rf_param_columns,
    )

    rf_max_depth_results = run_rf_max_depth_study(X_train, y_train, X_test, y_test)
    print_table(
        "6.2 RF - głębokość drzewa (max_depth)",
        rf_max_depth_results,
        columns=rf_param_columns,
    )

    best_rf = get_best_result(
        rf_n_estimators_results + rf_max_depth_results, metric_name="TEST_MAPE"
    )
    print_best_result("Najlepszy wariant Random Forest", best_rf)

    method_results.append(
        {
            "method": "Random Forest Best",
            "TRAIN_MSE": best_rf["TRAIN_MSE"],
            "TRAIN_RMSE": best_rf["TRAIN_RMSE"],
            "TRAIN_MAE": best_rf["TRAIN_MAE"],
            "TRAIN_MAPE": best_rf["TRAIN_MAPE"],
            "TEST_MSE": best_rf["TEST_MSE"],
            "TEST_RMSE": best_rf["TEST_RMSE"],
            "TEST_MAE": best_rf["TEST_MAE"],
            "TEST_MAPE": best_rf["TEST_MAPE"],
        }
    )

    # =========================
    # 7. Sieć neuronowa - pojedynczy wariant
    # =========================
    print_section("7. Sieć neuronowa - pojedynczy wariant")

    single_nn_evaluation = evaluate_neural_network_configuration(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        hidden_layers=[16],
        activation="tanh",
        learning_rate=0.01,
        epochs=2000,
        initialization="small_random",
        batch_size="full_batch",
        random_seed=42,
        early_stopping=True,
    )

    print_metrics(
        "Neural Network - TRAIN", y_train, single_nn_evaluation["y_pred_train"]
    )
    print(
        f"\nNeural Network - VALIDATION\nMAPE: "
        f"{single_nn_evaluation['validation_metrics']['MAPE']:.4f}"
    )
    print_metrics("Neural Network - TEST", y_test, single_nn_evaluation["y_pred_test"])

    # =========================
    # 7. Analiza parametrów sieci neuronowej
    # =========================
    print_section("8. Analiza parametrów sieci neuronowej")

    parameter_studies = run_all_neural_network_parameter_studies(
        file_path=data_file_path,
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
    )

    parameter_table_columns = [
        "tested_value",
        "Train Mean",
        "Train Std",
        "Train Min",
        "Train Max",
        "Test Mean",
        "Test Std",
        "Test Min",
        "Test Max",
    ]

    print_table(
        "7.1 Learning Rate Study (MAPE)",
        parameter_studies["learning_rate"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.2 Hidden Size Study (MAPE)",
        parameter_studies["hidden_size"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.3 Activation Study (MAPE)",
        parameter_studies["activation"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.4 Epochs Study (MAPE)",
        parameter_studies["epochs"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.5 Hidden Layers Study (MAPE)",
        parameter_studies["hidden_layers"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.6 Split Ratio Study (MAPE)",
        parameter_studies["split_ratio"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.7 Initialization Study (MAPE)",
        parameter_studies["initialization"],
        columns=parameter_table_columns,
    )
    print_table(
        "7.8 Batch Size Study (MAPE)",
        parameter_studies["batch_size"],
        columns=parameter_table_columns,
    )

    # =========================
    # 8. Finalny model SSN - pełne eksperymenty
    # =========================
    print_section("9. Finalny model SSN - pełne eksperymenty")

    nn_results = run_neural_network_experiments(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
    )

    best_nn = get_best_result(nn_results, metric_name="VAL_MAPE")
    print_best_result(
        "Najlepsza konfiguracja sieci neuronowej (wybór po VAL MAPE)", best_nn
    )

    # =========================
    # 9. Porównanie finalnych kandydatów NN
    # =========================
    print_section("10. Porównanie finalnych kandydatów NN")

    final_nn_candidates = run_final_neural_network_candidates(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
    )

    final_candidate_columns = [
        "hidden_size",
        "activation",
        "learning_rate",
        "epochs",
        "TRAIN_MAPE",
        "TEST_MAPE",
        "RMSE",
        "MAE",
    ]
    final_nn_candidates_sorted = sorted(
        final_nn_candidates, key=lambda result: result["TEST_MAPE"]
    )
    print_table(
        "Final Neural Network Candidate Comparison",
        final_nn_candidates_sorted,
        columns=final_candidate_columns,
    )

    best_nn_evaluation = evaluate_neural_network_configuration(
        X_train=X_train,
        y_train=y_train,
        y_train_scaled=y_train_scaled,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        X_test=X_test,
        y_test=y_test,
        target_scaling_params=target_scaling_params,
        hidden_layers=[best_nn["hidden_size"]],
        activation=best_nn["activation"],
        learning_rate=best_nn["learning_rate"],
        epochs=best_nn["epochs"],
        initialization=best_nn["initialization"],
        batch_size=best_nn["batch_size"],
        random_seed=42,
        early_stopping=True,
    )

    best_nn_train_metrics = print_metrics(
        "Best Neural Network - TRAIN",
        y_train,
        best_nn_evaluation["y_pred_train"],
    )
    print(
        f"\nBest Neural Network - VALIDATION\nMAPE: "
        f"{best_nn_evaluation['validation_metrics']['MAPE']:.4f}"
    )
    best_nn_test_metrics = print_metrics(
        "Best Neural Network - TEST",
        y_test,
        best_nn_evaluation["y_pred_test"],
    )
    method_results.append(
        build_method_result_row(
            "Neural Network Best", best_nn_train_metrics, best_nn_test_metrics
        )
    )

    # =========================
    # 10. Zapis wyników do plików
    # =========================
    print_section("11. Zapis wyników")

    method_results_sorted = sorted(
        method_results, key=lambda result: result["TEST_MAPE"]
    )

    save_rows_to_csv(
        f"{tables_dir}/main_methods_comparison.csv",
        method_results_sorted,
        columns=[
            "method",
            "TRAIN_MSE",
            "TRAIN_RMSE",
            "TRAIN_MAE",
            "TRAIN_MAPE",
            "TEST_MSE",
            "TEST_RMSE",
            "TEST_MAE",
            "TEST_MAPE",
        ],
    )
    save_rows_to_csv(
        f"{tables_dir}/knn_experiments.csv",
        knn_results_sorted,
        columns=knn_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/rf_n_estimators_study.csv",
        rf_n_estimators_results,
        columns=rf_param_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/rf_max_depth_study.csv",
        rf_max_depth_results,
        columns=rf_param_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/learning_rate_study.csv",
        parameter_studies["learning_rate"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/hidden_size_study.csv",
        parameter_studies["hidden_size"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/activation_study.csv",
        parameter_studies["activation"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/epochs_study.csv",
        parameter_studies["epochs"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/hidden_layers_study.csv",
        parameter_studies["hidden_layers"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/split_ratio_study.csv",
        parameter_studies["split_ratio"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/initialization_study.csv",
        parameter_studies["initialization"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/batch_size_study.csv",
        parameter_studies["batch_size"],
        columns=parameter_table_columns,
    )
    save_rows_to_csv(
        f"{tables_dir}/final_nn_candidate_comparison.csv",
        final_nn_candidates_sorted,
        columns=final_candidate_columns,
    )

    summary_text = build_summary_text(best_knn, best_nn, method_results_sorted)
    save_text_file(f"{text_dir}/summary.txt", summary_text)

    print("Zapisano pliki:")
    print(f"- {tables_dir}/main_methods_comparison.csv")
    print(f"- {tables_dir}/knn_experiments.csv")
    print(f"- {tables_dir}/rf_n_estimators_study.csv")
    print(f"- {tables_dir}/rf_max_depth_study.csv")
    print(f"- {tables_dir}/learning_rate_study.csv")
    print(f"- {tables_dir}/hidden_size_study.csv")
    print(f"- {tables_dir}/activation_study.csv")
    print(f"- {tables_dir}/epochs_study.csv")
    print(f"- {tables_dir}/hidden_layers_study.csv")
    print(f"- {tables_dir}/split_ratio_study.csv")
    print(f"- {tables_dir}/initialization_study.csv")
    print(f"- {tables_dir}/batch_size_study.csv")
    print(f"- {tables_dir}/final_nn_candidate_comparison.csv")
    print(f"- {text_dir}/summary.txt")


if __name__ == "__main__":
    main()
