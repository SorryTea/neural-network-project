"""
Plik zawiera implementacje modeli używanych w projekcie.

Zawarte modele:
1. BaselineLastValueRegressor
2. BaselineMovingAverageRegressor
3. LinearRegressionGD
4. KNNRegressor
5. NeuralNetworkRegressor

Wszystkie modele są zaimplementowane bez gotowych bibliotek ML.
"""

import numpy as np


class BaselineLastValueRegressor:
    """
    Model bazowy:
    przewiduje cenę na podstawie ostatniej znanej wartości Close_lag_1.
    """

    def __init__(self, close_lag_1_index=5):
        self.close_lag_1_index = close_lag_1_index

    def fit(self, X, y):
        """
        Baseline nie wymaga uczenia.
        """
        return self

    def predict(self, X):
        """
        Zwraca kolumnę Close_lag_1 jako predykcję.
        """
        X = np.array(X, dtype=float)
        return X[:, self.close_lag_1_index]


class BaselineMovingAverageRegressor:
    """
    Model bazowy:
    przewiduje cenę jako średnią z lagów Close.
    Domyślnie korzysta z Close_lag_1, Close_lag_2, Close_lag_3.
    """

    def __init__(self, lag_indices=None):
        if lag_indices is None:
            lag_indices = [5, 6, 7]
        self.lag_indices = lag_indices

    def fit(self, X, y):
        """
        Baseline nie wymaga uczenia.
        """
        return self

    def predict(self, X):
        """
        Zwraca średnią z wybranych lagów.
        """
        X = np.array(X, dtype=float)
        return np.mean(X[:, self.lag_indices], axis=1)


class LinearRegressionGD:
    """
    Własna implementacja regresji liniowej z użyciem gradient descent.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        """
        Uczy model regresji liniowej.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            error = y_pred - y

            dw = (2.0 / n_samples) * np.dot(X.T, error)
            db = (2.0 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(error**2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        """
        Zwraca predykcje modelu.
        """
        X = np.array(X, dtype=float)
        return np.dot(X, self.weights) + self.bias


class KNNRegressor:
    """
    Własna implementacja regresji k-NN.
    """

    def __init__(self, k=5, distance_metric="euclidean", weighted=False):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        k-NN nie uczy parametrów, tylko zapamiętuje dane treningowe.
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        return self

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2))

    def _minkowski_distance(self, x1, x2, p=3):
        return np.sum(np.abs(x1 - x2) ** p) ** (1.0 / p)

    def _calculate_distance(self, x1, x2):
        if self.distance_metric == "euclidean":
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == "manhattan":
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == "chebyshev":
            return self._chebyshev_distance(x1, x2)
        elif self.distance_metric == "minkowski":
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError("Nieobsługiwana metryka odległości.")

    def _predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            dist = self._calculate_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda pair: pair[0])
        nearest_neighbors = distances[: self.k]

        if not self.weighted:
            neighbor_values = [value for _, value in nearest_neighbors]
            return np.mean(neighbor_values)

        weighted_sum = 0.0
        weight_total = 0.0

        for dist, value in nearest_neighbors:
            if dist == 0:
                return value
            weight = 1.0 / dist
            weighted_sum += weight * value
            weight_total += weight

        return weighted_sum / weight_total

    def predict(self, X):
        """
        Zwraca predykcje dla wszystkich obserwacji.
        """
        X = np.array(X, dtype=float)
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions, dtype=float)


class NeuralNetworkRegressor:
    """
    Własna sieć neuronowa do regresji:
    - dowolna liczba warstw ukrytych,
    - liniowa warstwa wyjściowa,
    - backpropagation zaimplementowany ręcznie w numpy,
    - kilka sposobów inicjalizacji wag,
    - tryb full-batch albo mini-batch.
    """

    def __init__(
        self,
        input_size,
        hidden_size=16,
        hidden_layers=None,
        learning_rate=0.01,
        epochs=1000,
        activation="relu",
        initialization="small_random",
        batch_size="full_batch",
        random_seed=42,
        early_stopping_patience=200,
        min_delta=0.0,
    ):
        self.input_size = input_size
        self.hidden_layers = self._resolve_hidden_layers(hidden_size, hidden_layers)
        self.hidden_size = self.hidden_layers[0]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.initialization = initialization
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self.weights = []
        self.biases = []

        self.loss_history = []
        self.validation_loss_history = []
        self.best_epoch = None
        self.stopped_epoch = None

        self._initialize_parameters()

    def _resolve_hidden_layers(self, hidden_size, hidden_layers):
        if hidden_layers is not None:
            resolved_hidden_layers = list(hidden_layers)
        elif isinstance(hidden_size, (list, tuple)):
            resolved_hidden_layers = list(hidden_size)
        else:
            resolved_hidden_layers = [hidden_size]

        if len(resolved_hidden_layers) == 0:
            raise ValueError("Sieć musi mieć co najmniej jedną warstwę ukrytą.")

        return resolved_hidden_layers

    def _get_layer_sizes(self):
        return [self.input_size] + self.hidden_layers + [1]

    def _initialize_weight_matrix(self, fan_in, fan_out):
        if self.initialization == "zeros":
            return np.zeros((fan_in, fan_out))
        elif self.initialization == "small_random":
            return np.random.randn(fan_in, fan_out) * 0.01
        elif self.initialization == "xavier":
            return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
        elif self.initialization == "he":
            return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        else:
            raise ValueError("Nieobsługiwana metoda inicjalizacji wag.")

    def _initialize_parameters(self):
        """
        Inicjalizacja wag i biasów.
        """
        np.random.seed(self.random_seed)

        self.weights = []
        self.biases = []

        layer_sizes = self._get_layer_sizes()

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            self.weights.append(self._initialize_weight_matrix(fan_in, fan_out))
            self.biases.append(np.zeros((1, fan_out)))

    def _copy_parameters(self):
        copied_weights = [weight.copy() for weight in self.weights]
        copied_biases = [bias.copy() for bias in self.biases]
        return copied_weights, copied_biases

    def _restore_parameters(self, copied_weights, copied_biases):
        self.weights = [weight.copy() for weight in copied_weights]
        self.biases = [bias.copy() for bias in copied_biases]

    def _activate(self, Z):
        """
        Funkcja aktywacji dla warstw ukrytych.
        """
        if self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "leaky_relu":
            return np.where(Z > 0, Z, 0.01 * Z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation == "tanh":
            return np.tanh(Z)
        else:
            raise ValueError("Nieobsługiwana funkcja aktywacji.")

    def _activate_derivative(self, Z):
        """
        Pochodna funkcji aktywacji.
        """
        if self.activation == "relu":
            return (Z > 0).astype(float)
        elif self.activation == "leaky_relu":
            return np.where(Z > 0, 1.0, 0.01)
        elif self.activation == "sigmoid":
            sigmoid = 1.0 / (1.0 + np.exp(-Z))
            return sigmoid * (1.0 - sigmoid)
        elif self.activation == "tanh":
            return 1.0 - np.tanh(Z) ** 2
        else:
            raise ValueError("Nieobsługiwana funkcja aktywacji.")

    def _forward(self, X):
        """
        Forward propagation dla wielu warstw.
        """
        A_values = [X]
        Z_values = []

        A = X

        for layer_index in range(len(self.hidden_layers)):
            Z = np.dot(A, self.weights[layer_index]) + self.biases[layer_index]
            A = self._activate(Z)

            Z_values.append(Z)
            A_values.append(A)

        Z_output = np.dot(A, self.weights[-1]) + self.biases[-1]
        y_pred = Z_output

        Z_values.append(Z_output)
        A_values.append(y_pred)

        cache = {"A_values": A_values, "Z_values": Z_values}

        return y_pred, cache

    def _compute_loss(self, y_true, y_pred):
        """
        MSE
        """
        y_true = y_true.reshape(-1, 1)
        return np.mean((y_true - y_pred) ** 2)

    def _backward(self, y_true, cache):
        """
        Backpropagation dla wielu warstw.
        """
        y_true = y_true.reshape(-1, 1)

        A_values = cache["A_values"]
        Z_values = cache["Z_values"]

        layer_count = len(self.weights)
        sample_count = y_true.shape[0]

        gradients_W = [None] * layer_count
        gradients_b = [None] * layer_count

        dZ = (2.0 / sample_count) * (A_values[-1] - y_true)

        for layer_index in range(layer_count - 1, -1, -1):
            A_prev = A_values[layer_index]

            gradients_W[layer_index] = np.dot(A_prev.T, dZ)
            gradients_b[layer_index] = np.sum(dZ, axis=0, keepdims=True)

            if layer_index > 0:
                dA_prev = np.dot(dZ, self.weights[layer_index].T)
                dZ = dA_prev * self._activate_derivative(Z_values[layer_index - 1])

        return {"dW": gradients_W, "db": gradients_b}

    def _update_parameters(self, gradients):
        """
        Aktualizacja wag.
        """
        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= self.learning_rate * gradients["dW"][layer_index]
            self.biases[layer_index] -= self.learning_rate * gradients["db"][layer_index]

    def _resolve_batch_size(self, n_samples):
        if self.batch_size in [None, "full_batch"]:
            return n_samples

        resolved_batch_size = int(self.batch_size)

        if resolved_batch_size <= 0:
            return n_samples

        return min(resolved_batch_size, n_samples)

    def _iterate_batches(self, X, y):
        n_samples = X.shape[0]
        resolved_batch_size = self._resolve_batch_size(n_samples)

        for start_index in range(0, n_samples, resolved_batch_size):
            end_index = min(start_index + resolved_batch_size, n_samples)
            yield X[start_index:end_index], y[start_index:end_index]

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        verbose=False,
        print_every=100,
        early_stopping=True,
    ):
        """
        Uczenie sieci.
        Jeśli podano dane walidacyjne, można użyć prostego early stopping.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        use_validation = X_val is not None and y_val is not None

        if use_validation:
            X_val = np.array(X_val, dtype=float)
            y_val = np.array(y_val, dtype=float)

        self._initialize_parameters()
        self.loss_history = []
        self.validation_loss_history = []
        self.best_epoch = None
        self.stopped_epoch = None

        best_validation_loss = np.inf
        best_weights = None
        best_biases = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            epoch_loss_sum = 0.0
            epoch_sample_count = 0

            for X_batch, y_batch in self._iterate_batches(X, y):
                y_pred_batch, cache = self._forward(X_batch)
                batch_loss = self._compute_loss(y_batch, y_pred_batch)

                gradients = self._backward(y_batch, cache)
                self._update_parameters(gradients)

                epoch_loss_sum += batch_loss * len(X_batch)
                epoch_sample_count += len(X_batch)

            epoch_loss = epoch_loss_sum / epoch_sample_count
            self.loss_history.append(epoch_loss)

            validation_loss = None

            if use_validation:
                validation_predictions, _ = self._forward(X_val)
                validation_loss = self._compute_loss(y_val, validation_predictions)
                self.validation_loss_history.append(validation_loss)

                if early_stopping:
                    if validation_loss < best_validation_loss - self.min_delta:
                        best_validation_loss = validation_loss
                        best_weights, best_biases = self._copy_parameters()
                        self.best_epoch = epoch + 1
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                        if epochs_without_improvement >= self.early_stopping_patience:
                            self.stopped_epoch = epoch + 1
                            break

            if verbose and (epoch + 1) % print_every == 0:
                if validation_loss is None:
                    print(f"Epoka {epoch + 1}/{self.epochs}, loss = {epoch_loss:.6f}")
                else:
                    print(
                        f"Epoka {epoch + 1}/{self.epochs}, "
                        f"loss = {epoch_loss:.6f}, val_loss = {validation_loss:.6f}"
                    )

        if use_validation and early_stopping and best_weights is not None:
            self._restore_parameters(best_weights, best_biases)

        if self.stopped_epoch is None:
            self.stopped_epoch = len(self.loss_history)

        return self

    def predict(self, X):
        """
        Predykcja.
        """
        X = np.array(X, dtype=float)
        y_pred, _ = self._forward(X)
        return y_pred.flatten()
