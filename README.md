# 🧠 MLP Neural Network From Scratch

A comprehensive, educational implementation of Multi-Layer Perceptron (MLP) neural networks built entirely from scratch using NumPy. This project provides a deep understanding of neural network fundamentals, backpropagation, and various optimization techniques.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)


## 🎯 Features

### Core Functionality
- **Complete MLP Implementation**: Full forward and backward propagation from scratch
- **Flexible Architecture**: Support for any number of hidden layers and neurons
- **Multiple Activation Functions**: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax, Linear
- **Various Loss Functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Smart Weight Initialization**: He initialization for ReLU, Xavier for others
- **Gradient Clipping**: Prevents exploding gradients during training
- **Numerical Gradient Checking**: Built-in gradient verification for debugging

### Specialized Classes
- **MLPClassifier**: Optimized for classification tasks (binary & multi-class)
- **MLPRegressor**: Specialized for regression problems
- **Automatic Architecture**: Dynamic layer configuration based on data

### Utilities & Tools
- **Data Generation**: Synthetic datasets for testing and experimentation
- **Preprocessing**: Data normalization and train/test/validation splitting
- **Visualization**: Decision boundary plots and training history graphs
- **One-Hot Encoding**: Built-in categorical label encoding

## 📦 Installation

### Option 1: Clone and Install
```bash
git clone https://github.com/hghaemi/mlp_neural_network_from_scratch.git
cd mlp_neural_network_from_scratch
pip install -r requirements.txt
```

### Option 2: Development Installation
```bash
git clone https://github.com/hghaemi/mlp_neural_network_from_scratch.git
cd mlp_neural_network_from_scratch
pip install -e .
```

### Requirements
- Python 3.7+
- NumPy >= 1.20.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0 (for visualization)
- jupyter >= 1.0.0 (for notebooks)

## 🚀 Quick Start

### Binary Classification
```python
import numpy as np
from mlp import MLPClassifier
from mlp.utils import generate_classification_data, normalize_data

# Generate data
X, y = generate_classification_data(n_samples=1000, n_features=2, n_classes=2)
X_normalized, scaler = normalize_data(X)

# Create and train model
mlp = MLPClassifier(
    hidden_layers=[10, 5],
    activation='relu',
    learning_rate=0.01,
    random_state=42
)

mlp.fit(X_normalized, y, epochs=1000, verbose=True)

# Make predictions
predictions = mlp.predict_classes(X_normalized)
accuracy = np.mean(predictions.flatten() == y)
print(f"Accuracy: {accuracy:.4f}")
```

### Multi-class Classification
```python
from mlp.utils import one_hot_encode

# Generate multi-class data
X, y = generate_classification_data(n_samples=1000, n_features=2, n_classes=3)
y_onehot = one_hot_encode(y, n_classes=3)

# Create classifier with softmax output
mlp = MLPClassifier(
    hidden_layers=[20, 15, 10],
    activation='relu',
    output_activation='softmax',
    learning_rate=0.01
)

mlp.fit(X, y_onehot, epochs=1500)
```

### Regression
```python
from mlp import MLPRegressor
from mlp.utils import generate_regression_data

# Generate regression data
X, y = generate_regression_data(n_samples=1000, n_features=1, noise=10)

# Create regressor
mlp = MLPRegressor(
    hidden_layers=[50, 30, 10],
    activation='relu',
    learning_rate=0.001
)

mlp.fit(X, y, epochs=2000)

# Evaluate
predictions = mlp.predict(X)
mse = np.mean((y - predictions.flatten()) ** 2)
print(f"MSE: {mse:.4f}")
```

## 📊 Advanced Usage

### Custom Architecture with Low-Level MLP
```python
from mlp import MLP

# Define custom architecture
mlp = MLP(
    layers=[4, 16, 8, 3],  # 4 inputs, two hidden layers, 3 outputs
    activations=['relu', 'relu', 'softmax'],
    loss='categorical_crossentropy',
    learning_rate=0.01
)

# Train with custom data
mlp.fit(X_train, y_train, epochs=1000)
```

### Gradient Checking (for debugging)
```python
# Verify gradients numerically
mlp = MLP(layers=[2, 5, 1], activations=['relu', 'sigmoid'], loss='mse')
is_correct = mlp.check_gradients(X_small, y_small)
print(f"Gradients correct: {is_correct}")
```

### Training History and Visualization
```python
from mlp.utils import plot_training_history, plot_decision_boundary

# Train model
mlp.fit(X, y, epochs=1000)

# Plot training progress
plot_training_history(mlp.history, "Training Progress")

# Visualize decision boundary (2D data only)
plot_decision_boundary(mlp, X, y)
```

## 🏗️ Architecture

### Project Structure
```
mlp_neural_network_from_scratch/
├── examples/
│   ├── basic_example.py           # Binary classification demo
│   ├── multiclass_example.py      # Multi-class classification
│   ├── regression_example.py      # Regression demo
│   └── visualization_demo.ipynb   # Jupyter notebook with visualizations
├── mlp/
│   ├── __init__.py              # Package initialization
│   ├── activations.py           # Activation functions
│   ├── losses.py                # Loss functions
│   ├── models.py                # MLP, MLPClassifier, MLPRegressor classes
│   └── utils.py                 # Utilities and data generation helpers
├── tests/
│   ├── __init__.py              # Package initialization
│   ├── test_activations.py      # Unit tests for activation functions
│   ├── test_losses.py           # Unit tests for loss functions
│   └── test_models.py           # Unit tests for model classes
├── .gitignore                   # Git ignore rules
├── LICENSE                      # Project license
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation script
```

### Core Components

#### 1. Neural Network Models (`models.py`)
- **MLP**: Base class with full neural network implementation
- **MLPClassifier**: Specialized for classification with automatic architecture
- **MLPRegressor**: Optimized for regression tasks

#### 2. Activation Functions (`activations.py`)
```python
Available activations:
- 'relu': Rectified Linear Unit
- 'sigmoid': Sigmoid activation
- 'tanh': Hyperbolic tangent
- 'leaky_relu': Leaky ReLU with small negative slope
- 'softmax': Softmax for multi-class classification
- 'linear': Linear activation (no transformation)
```

#### 3. Loss Functions (`losses.py`)
```python
Available losses:
- 'mse': Mean Squared Error (regression)
- 'binary_crossentropy': Binary classification
- 'categorical_crossentropy': Multi-class classification
```

#### 4. Utilities (`utils.py`)
- Data generation functions
- Preprocessing tools
- Visualization helpers
- Train/test/validation splitting

## 🧮 Mathematical Foundation

### Forward Propagation
For each layer `l`:
```
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = activation(z^(l))
```

### Backpropagation
Gradient computation using chain rule:
```python
# Output layer
dz^(L) = ∇_a C ⊙ activation'(z^(L))

# Hidden layers
dz^(l) = (W^(l+1))^T * dz^(l+1) ⊙ activation'(z^(l))

# Parameter gradients
dW^(l) = dz^(l) * (a^(l-1))^T
db^(l) = dz^(l)
```

### Weight Initialization
- **He Initialization** (for ReLU): `W ~ N(0, sqrt(2/n_in))`
- **Xavier Initialization** (for others): `W ~ N(0, sqrt(1/n_in))`

## 🔧 Configuration Options

### MLPClassifier Parameters
```python
MLPClassifier(
    hidden_layers=[100],        # List of hidden layer sizes
    activation='relu',          # Hidden layer activation
    output_activation='sigmoid', # Output activation
    learning_rate=0.01,        # Learning rate
    random_state=None          # Random seed
)
```

### MLPRegressor Parameters
```python
MLPRegressor(
    hidden_layers=[100],       # List of hidden layer sizes
    activation='relu',         # Hidden layer activation
    learning_rate=0.01,       # Learning rate
    random_state=None         # Random seed
)
```

### Training Parameters
```python
model.fit(
    X, y,                     # Training data
    epochs=1000,              # Number of training epochs
    verbose=True,             # Print training progress
    validation_data=None      # Optional validation set
)
```

## 📈 Performance Tips

### 1. Learning Rate Selection
- **Too high**: Loss may oscillate or diverge
- **Too low**: Slow convergence
- **Recommended**: Start with 0.01, adjust based on loss behavior

### 2. Architecture Design
- **Start simple**: Begin with 1-2 hidden layers
- **Increase gradually**: Add layers/neurons if underfitting
- **Monitor overfitting**: Use validation data

### 3. Data Preprocessing
```python
# Always normalize input data
X_normalized, scaler = normalize_data(X, method='standard')

# For classification, ensure proper label format
y_onehot = one_hot_encode(y, n_classes=num_classes)
```

### 4. Activation Function Guidelines
- **ReLU**: Good default for hidden layers
- **Sigmoid**: Binary classification output
- **Softmax**: Multi-class classification output
- **Linear**: Regression output

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_models.py

# Run with coverage
pytest --cov=mlp tests/
```

### Test Coverage
- Model initialization and architecture
- Forward and backward propagation
- Training convergence
- Prediction accuracy
- Gradient correctness

## 📚 Examples

### Complete Examples
The `examples/` directory contains full implementations:

1. **Binary Classification** (`basic_example.py`)
   - Synthetic 2D dataset
   - Visualization of decision boundary
   - Training history plots

2. **Multi-class Classification** (`multiclass_example.py`)
   - Spiral dataset with 3 classes
   - Softmax output layer
   - One-hot encoding

3. **Regression** (`regression_example.py`)
   - 1D regression with noise
   - Multiple hidden layers
   - MSE evaluation

4. **Interactive Notebook** (`visualization_demo.ipynb`)
   - Step-by-step walkthrough
   - Interactive visualizations
   - Parameter experimentation

### Running Examples
```bash
# Binary classification
python examples/basic_example.py

# Multi-class classification
python examples/multiclass_example.py

# Regression
python examples/regression_example.py

# Jupyter notebook
jupyter notebook examples/visualization_demo.ipynb
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Poor Convergence
**Symptoms**: Loss not decreasing, low accuracy
**Solutions**:
- Reduce learning rate
- Increase number of epochs
- Add more hidden layers/neurons
- Check data normalization

#### 2. Overfitting
**Symptoms**: Training accuracy >> test accuracy
**Solutions**:
- Reduce model complexity
- Use validation data for early stopping
- Add more training data

#### 3. Exploding Gradients
**Symptoms**: Loss becomes NaN, very large gradient norms
**Solutions**:
- Reduce learning rate
- Gradient clipping is built-in (max_grad_norm=5.0)
- Check data scaling

#### 4. Vanishing Gradients
**Symptoms**: Very slow learning, especially in deep networks
**Solutions**:
- Use ReLU activation instead of sigmoid/tanh
- Proper weight initialization (He/Xavier)
- Consider fewer layers

### Debugging Tools
```python
# Check gradient correctness
mlp.check_gradients(X_sample, y_sample)

# Monitor gradient norms
mlp.fit(X, y, epochs=100, verbose=True)  # Shows initial gradient norm

# Examine training history
import matplotlib.pyplot as plt
plt.plot(mlp.history['loss'])
plt.title('Training Loss')
plt.show()
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Ensure tests pass**: `python -m pytest tests/`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Create Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include type hints where appropriate
- Write comprehensive tests
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NumPy Community** for the fundamental array operations
- **scikit-learn** for inspiration on API design
- **Deep Learning Literature** for mathematical foundations
- **Open Source Community** for tools and best practices

## 📬 Contact

**M. Hossein Ghaemi**
- Email: h.ghaemi.2003@gmail.com
- GitHub: [@hghaemi](https://github.com/hghaemi)
- Project Link: [https://github.com/hghaemi/mlp_neural_network_from_scratch.git](https://github.com/hghaemi/mlp_neural_network_from_scratch.git)
