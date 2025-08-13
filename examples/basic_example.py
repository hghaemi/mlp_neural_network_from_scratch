import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mlp import MLPClassifier
from mlp.utils import generate_classification_data, normalize_data, train_test_validation_split, plot_decision_boundary, plot_training_history


def main():
    print("=== MLP Binary Classification Example ===\n")
    
    print("1. Generating synthetic binary classification data...")
    X, y = generate_classification_data(n_samples=1000, n_features=2, n_classes=2, random_state=42)
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
    
    print("\n4. Creating and training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layers=[10, 5],
        activation='relu',
        output_activation='sigmoid',
        learning_rate=0.01,
        random_state=42
    )
    
    mlp.fit(X_train, y_train, epochs=1000, verbose=True)
    
    print("\n5. Making predictions...")
    train_predictions = mlp.predict_classes(X_train)
    test_predictions = mlp.predict_classes(X_test)
    
    train_accuracy = np.mean(train_predictions.flatten() == y_train)
    test_accuracy = np.mean(test_predictions.flatten() == y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\n6. Plotting results...")
    plot_decision_boundary(mlp, X_normalized, y)
    plot_training_history(mlp.history, "Binary Classification Training")


if __name__ == "__main__":
    main()