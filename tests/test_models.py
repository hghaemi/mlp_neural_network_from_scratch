import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import MLP, MLPClassifier, MLPRegressor
from mlp.utils import generate_classification_data, generate_regression_data, one_hot_encode


class TestMLP(unittest.TestCase):
    
    def setUp(self):
        self.X_class, self.y_class = generate_classification_data(
            n_samples=100, n_features=2, n_classes=2, random_state=42
        )
        self.X_reg, self.y_reg = generate_regression_data(
            n_samples=100, n_features=2, random_state=42
        )
        
    def test_mlp_initialization(self):
        mlp = MLP(layers=[2, 5, 1], activations=['relu', 'sigmoid'], 
                  loss='mse', learning_rate=0.01, random_state=42)
        
        self.assertEqual(len(mlp.weights), 2)
        self.assertEqual(len(mlp.biases), 2)
        self.assertEqual(mlp.weights[0].shape, (2, 5))
        self.assertEqual(mlp.weights[1].shape, (5, 1))
        
    def test_forward_pass(self):
        mlp = MLP(layers=[2, 5, 1], activations=['relu', 'sigmoid'], 
                  loss='mse', learning_rate=0.01, random_state=42)
        
        activations = mlp.forward(self.X_class)
        
        self.assertEqual(len(activations), 3)  # input + 2 layers
        self.assertEqual(activations[0].shape, self.X_class.shape)
        self.assertEqual(activations[-1].shape, (self.X_class.shape[0], 1))
        
    def test_backward_pass(self):
        mlp = MLP(layers=[2, 5, 1], activations=['relu', 'sigmoid'], 
                  loss='mse', learning_rate=0.01, random_state=42)
        
        activations = mlp.forward(self.X_class)
        dW, db = mlp.backward(self.X_class, self.y_class.reshape(-1, 1), activations)
        
        self.assertEqual(len(dW), 2)
        self.assertEqual(len(db), 2)
        self.assertEqual(dW[0].shape, mlp.weights[0].shape)
        self.assertEqual(dW[1].shape, mlp.weights[1].shape)
        
    def test_training(self):
        mlp = MLP(layers=[2, 5, 1], activations=['relu', 'sigmoid'], 
                  loss='binary_crossentropy', learning_rate=0.1, random_state=42)
        
        initial_weights = [w.copy() for w in mlp.weights]
        mlp.fit(self.X_class, self.y_class.reshape(-1, 1), epochs=10, verbose=False)
        
        weights_changed = any(not np.allclose(initial_weights[i], mlp.weights[i]) 
                             for i in range(len(mlp.weights)))
        self.assertTrue(weights_changed)
        
    def test_predictions(self):
        mlp = MLP(layers=[2, 5, 1], activations=['relu', 'sigmoid'], 
                  loss='binary_crossentropy', learning_rate=0.1, random_state=42)
        
        mlp.fit(self.X_class, self.y_class.reshape(-1, 1), epochs=10, verbose=False)
        predictions = mlp.predict(self.X_class)
        class_predictions = mlp.predict_classes(self.X_class)
        
        self.assertEqual(predictions.shape, (self.X_class.shape[0], 1))
        self.assertEqual(class_predictions.shape, (self.X_class.shape[0], 1))
        self.assertTrue(np.all((class_predictions == 0) | (class_predictions == 1)))


class TestMLPClassifier(unittest.TestCase):
    
    def setUp(self):
        self.X_binary, self.y_binary = generate_classification_data(
            n_samples=200, n_features=2, n_classes=2, random_state=42
        )
        self.X_multi, self.y_multi = generate_classification_data(
            n_samples=200, n_features=2, n_classes=3, random_state=42
        )
        self.y_multi_onehot = one_hot_encode(self.y_multi, n_classes=3)
        
    def test_binary_classification(self):
        clf = MLPClassifier(hidden_layers=[10], activation='relu', 
                           learning_rate=0.1, random_state=42)
        
        clf.fit(self.X_binary, self.y_binary, epochs=100, verbose=False)
        predictions = clf.predict_classes(self.X_binary)
        accuracy = np.mean(predictions.flatten() == self.y_binary)
        
        self.assertGreater(accuracy, 0.7)  # Should achieve reasonable accuracy
        
    def test_multiclass_classification(self):
        clf = MLPClassifier(hidden_layers=[20, 10], activation='relu',
                           output_activation='softmax', learning_rate=0.01, random_state=42)
        
        clf.fit(self.X_multi, self.y_multi_onehot, epochs=200, verbose=False)
        predictions = clf.predict_classes(self.X_multi)
        accuracy = np.mean(predictions == self.y_multi)
        
        self.assertGreater(accuracy, 0.6)  # Should achieve reasonable accuracy


class TestMLPRegressor(unittest.TestCase):
    
    def setUp(self):
        self.X_reg, self.y_reg = generate_regression_data(
            n_samples=200, n_features=2, noise=0.1, random_state=42
        )
        
    def test_regression(self):
        reg = MLPRegressor(hidden_layers=[20, 10], activation='relu', 
                          learning_rate=0.001, random_state=42)
        
        reg.fit(self.X_reg, self.y_reg, epochs=200, verbose=False)
        predictions = reg.predict(self.X_reg)
        mse = np.mean((self.y_reg - predictions.flatten()) ** 2)
        
        self.assertLess(mse, 1000)


if __name__ == '__main__':
    unittest.main()