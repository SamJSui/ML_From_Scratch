import numpy as np
import matplotlib.pyplot as plt # PLOTTING
# DATA GENERATION
from sklearn.datasets import make_regression, make_blobs, load_breast_cancer
from LinearRegression import LinearRegression # MODELS
from SingleLayerPerceptron import SingleLayerPerceptron

def test_LinearRegression():
    X, Y = make_regression(n_samples=10, n_features=2, noise=2, random_state=42)
    Y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)

def test_SingleLayerPerceptron():
    data = load_breast_cancer()
    data, target = data['data'], data['target']
    print(data, target)

if __name__ == '__main__':
    test_SingleLayerPerceptron()