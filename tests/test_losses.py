import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp.losses import (
    mean_squared_error, mse_derivative,
    binary_cross_entropy, binary_cross_entropy_derivative,
    categorical_cross_entropy, categorical_cross_entropy_derivative,
    get_loss_function
)


class TestLossFunctions(unittest.TestCase):
    
    def setUp(self):
        self.y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
        self.y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8])
        
        self.y_true_binary = np.array([0, 1, 0, 1])
        self.y_pred_binary = np.array([0.1, 0.9, 0.2, 0.8])
        
        self.y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.y_pred_multi = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
        
    def test_mean_squared_error(self):
        result = mean_squared_error(self.y_true_reg, self.y_pred_reg)
        
        expected = np.mean((self.y_true_reg - self.y_pred_reg) ** 2)
        
        self.assertAlmostEqual(result, expected, places=7)
        self.assertGreaterEqual(result, 0)  # MSE should be non-negative
        
    def test_mse_derivative(self):
        result = mse_derivative(self.y_true_reg, self.y_pred_reg)
        
        expected = 2 * (self.y_pred_reg - self.y_true_reg) / len(self.y_true_reg)
        
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_binary_cross_entropy(self):
        result = binary_cross_entropy(self.y_true_binary, self.y_pred_binary)
        
        self.assertGreaterEqual(result, 0)  # BCE should be non-negative
        self.assertIsInstance(result, (float, np.floating))
        
        perfect_pred = self.y_true_binary.astype(float)
        perfect_pred[perfect_pred == 0] = 1e-15  # Avoid log(0)
        perfect_pred[perfect_pred == 1] = 1 - 1e-15  # Avoid log(0)
        
        perfect_loss = binary_cross_entropy(self.y_true_binary, perfect_pred)
        self.assertLess(perfect_loss, result)  
        
    def test_binary_cross_entropy_derivative(self):
        result = binary_cross_entropy_derivative(self.y_true_binary, self.y_pred_binary)
        
        self.assertEqual(result.shape, self.y_pred_binary.shape)
        
    def test_categorical_cross_entropy(self):
        result = categorical_cross_entropy(self.y_true_multi, self.y_pred_multi)
        
        self.assertGreaterEqual(result, 0)  # CCE should be non-negative
        self.assertIsInstance(result, (float, np.floating))
        
    def test_categorical_cross_entropy_derivative(self):
        result = categorical_cross_entropy_derivative(self.y_true_multi, self.y_pred_multi)
        
        self.assertEqual(result.shape, self.y_pred_multi.shape)
        
    def test_get_loss_function(self):
        loss_func, loss_deriv = get_loss_function('mse')
        self.assertEqual(loss_func, mean_squared_error)
        self.assertEqual(loss_deriv, mse_derivative)
        
        with self.assertRaises(ValueError):
            get_loss_function('invalid_loss')
            
    def test_loss_consistency(self):
        epsilon = 1e-7
        
        y_pred_plus = self.y_pred_reg + epsilon
        y_pred_minus = self.y_pred_reg - epsilon
        
        numerical_grad = (mean_squared_error(self.y_true_reg, y_pred_plus) - 
                         mean_squared_error(self.y_true_reg, y_pred_minus)) / (2 * epsilon)
        
        analytical_grad = np.mean(mse_derivative(self.y_true_reg, self.y_pred_reg))
        
        self.assertAlmostEqual(numerical_grad, analytical_grad, places=5)


if __name__ == '__main__':
    unittest.main()