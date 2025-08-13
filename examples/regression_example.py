import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mlp import MLPRegressor
from mlp.utils import generate_regression_data, normalize_data, train_test_validation_split, plot_training_history


def plot_regression_results(X, y, y_pred, title="Regression Results"):
    try:
        import matplotlib.pyplot as plt
        
        if X.shape[1] == 1:
            plt.figure(figsize=(10, 6))
            
            sort_idx = np.argsort(X.flatten())
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
            y_pred_sorted = y_pred[sort_idx]
            
            plt.scatter(X, y, alpha=0.6, label='True data')
            plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Prediction')
            plt.xlabel('Feature')
            plt.ylabel('Target')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, alpha=0.6)
            
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(title)
            plt.grid(True)
            plt.show()
            
    except ImportError:
        print("Matplotlib not available. Cannot plot regression results.")


def main():
    print("=== MLP Regression Example ===\n")
    
    print("1. Generating synthetic regression data...")
    X, y = generate_regression_data(n_samples=1000, n_features=1, noise=10, random_state=42)
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\n2. Normalizing the data...")
    X_normalized, scaler = normalize_data(X, method='standard')
    
    print("\n3. Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
        X_normalized, y, test_size=0.2, val_size=0.2, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\n4. Creating and training MLP regressor...")
    mlp = MLPRegressor(
        hidden_layers=[50, 30, 10],
        activation='relu',
        learning_rate=0.001,
        random_state=42
    )
    
    mlp.fit(X_train, y_train, epochs=2000, verbose=True)
    
    print("\n5. Making predictions...")
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)
    
    train_mse = np.mean((y_train - train_predictions.flatten()) ** 2)
    test_mse = np.mean((y_test - test_predictions.flatten()) ** 2)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    print(f"\nResults:")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    print("\n6. Plotting results...")
    plot_regression_results(X_test, y_test, test_predictions.flatten(), "Test Set Predictions")
    plot_training_history(mlp.history, "Regression Training")


if __name__ == "__main__":
    main()