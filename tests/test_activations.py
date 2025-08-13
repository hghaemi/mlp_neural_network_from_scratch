import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp.activations import (
    sigmoid, sigmoid_derivative, relu, relu_derivative,
    tanh, tanh_derivative, leaky_relu, leaky_relu_derivative,
    softmax, softmax_derivative, linear, linear_derivative,
    get_activation_function
)


class TestActivationFunctions(unittest.TestCase):
    
    def setUp(self):
        self.x = np.array([-2, -1, 0, 1, 2])
        self.x_2d = np.array([[-2, -1], [0, 1], [1, 2]])
        
    def test_sigmoid(self):
        result = sigmoid(self.x)
        
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))
        
        self.assertAlmostEqual(sigmoid(0), 0.5, places=7)
        
    def test_sigmoid_derivative(self):
        result = sigmoid_derivative(self.x)
        
        self.assertTrue(np.all(result >= 0))
        
        expected = 0.25  # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5
        self.assertAlmostEqual(sigmoid_derivative(0), expected, places=7)
        
    def test_relu(self):
        result = relu(self.x)
        expected = np.array([0, 0, 0, 1, 2])
        
        np.testing.assert_array_equal(result, expected)
        
    def test_relu_derivative(self):
        result = relu_derivative(self.x)
        expected = np.array([0, 0, 0, 1, 1])
        
        np.testing.assert_array_equal(result, expected)
        
    def test_tanh(self):
        result = tanh(self.x)
        expected = np.tanh(self.x)
        
        np.testing.assert_array_almost_equal(result, expected)
        
        self.assertTrue(np.all(result > -1))
        self.assertTrue(np.all(result < 1))
        
    def test_tanh_derivative(self):
        result = tanh_derivative(self.x)
        expected = 1 - np.tanh(self.x) ** 2
        
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_leaky_relu(self):
        alpha = 0.01
        result = leaky_relu(self.x, alpha=alpha)
        expected = np.where(self.x > 0, self.x, alpha * self.x)
        
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_leaky_relu_derivative(self):
        alpha = 0.01
        result = leaky_relu_derivative(self.x, alpha=alpha)
        expected = np.where(self.x > 0, 1.0, alpha)
        
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_softmax(self):
        result = softmax(self.x_2d)
        
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(self.x_2d.shape[0]))
        
        self.assertTrue(np.all(result > 0))
        
    def test_linear(self):
        result = linear(self.x)
        
        np.testing.assert_array_equal(result, self.x)
        
    def test_linear_derivative(self):
        result = linear_derivative(self.x)
        expected = np.ones_like(self.x)
        
        np.testing.assert_array_equal(result, expected)
        
    def test_get_activation_function(self):
        act_func, act_deriv = get_activation_function('relu')
        self.assertEqual(act_func, relu)
        self.assertEqual(act_deriv, relu_derivative)
        
        with self.assertRaises(ValueError):
            get_activation_function('invalid_activation')


if __name__ == '__main__':
    unittest.main()