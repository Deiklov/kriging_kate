import numpy as np
from collections import Counter


class KNN_classifier:
    def __init__(self, n_neighbors: int, **kwargs):
        self.K = n_neighbors

    def fit(self, x: np.array, y: np.array):
        self.X_train = x
        self.y_train = y

    def _predict(self, x):
        distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.K]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, x: np.array):
        predictions = [self._predict(el) for el in x]
        return np.array(predictions)
