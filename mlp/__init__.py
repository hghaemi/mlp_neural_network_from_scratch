"""
Multi-Layer Perceptron Neural Network Package
"""

from .models import MLP, MLPClassifier, MLPRegressor
from .activations import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    tanh, tanh_derivative,
    leaky_relu, leaky_relu_derivative,
    softmax, softmax_derivative,
    linear, linear_derivative
)
from .losses import (
    mean_squared_error, mse_derivative,
    binary_cross_entropy, binary_cross_entropy_derivative,
    categorical_cross_entropy, categorical_cross_entropy_derivative
)
from .utils import generate_classification_data, generate_regression_data, normalize_data

__version__ = "1.0.0"
__author__ = "M. Hossein Ghaemi"