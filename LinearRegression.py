import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class LinearRegression:
    def __init__(self):
        pass
        
    def _initialize_weights(self, X):
        self.n_samples, self.n_features = X.shape
        self.weights = np.random.randn(self.n_features) * 0.1 # weights
        self.bias = np.random.randint(0, 100)

    def _loss_fn(self, X, Y):
        y_hat = np.sum(self.weights * X, axis=1) + self.bias
        error = np.power((y_hat - Y), 2)
        cost = np.sum(error)
        return 1 / (2 * self.n_samples) * cost

    def predict(self, x):
        return np.dot(self.weights, x) + self.bias

    def _gradient(self, X, Y):
        dj_dw = np.zeros(self.n_features)
        dj_db = 0
        y_hat = np.sum(self.weights * X, axis=1) + self.bias
        error = y_hat-Y
        for j in range(self.n_features):
            dj_dw[j] = sum(error * X[:, j])
        dj_db = sum(error)
        return dj_dw / self.n_samples, dj_db / self.n_samples

    def fit(self, X, Y, epochs=100, verbose=False, lr=0.01):
        '''training linear model using gradient descent
            args:
                X (np.ndarray), input
                Y (np.ndarray), output
            optional args:
                epochs (int), number of iterations
                print_cost (bool), prints cost
                lr (float64), learning rate
            returns:
        '''
        J_history = []
        p_history = []
        self._initialize_weights(X)
        for i in range(0, epochs):
            cost = self._loss_fn(X, Y)
            dj_dw, dj_db = self._gradient(X, Y)
            for j in range(self.n_features):
                self.weights[j] = self.weights[j] - lr * dj_dw[j]
            self.bias = self.bias - lr * dj_db
            J_history.append(cost)
            p_history.append([self.weights, self.bias])
            if verbose and i % 100 == 0:
                print(f"Cost after iteration {i}:{cost : .2f}")
        self.J_history = J_history
        self.p_history = p_history
    