import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mlp import MLPClassifier
from mlp.utils import generate_spiral_data, normalize_data, one_hot_encode, train_test_validation_split, plot_decision_boundary, plot_training_history


def main():
    print("=== MLP Multi-class Classification Example ===\n")
    
    print("1. Generating synthetic spiral classification data...")
    X, y = generate_spiral_data(n_samples=500, n_classes=3, random_state=42)
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    print("\n2. Normalizing the data...")
    X_normalized, scaler = normalize_data(X, method='standard')
    
    print("\n3. Converting labels to one-hot encoding...")
    y_one_hot = one_hot_encode(y, n_classes=3)
    print(f"One-hot encoded labels shape: {y_one_hot.shape}")
    
    print("\n4. Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
        X_normalized, y_one_hot, test_size=0.2, val_size=0.2, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\n5. Creating and training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layers=[20, 15, 10],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01,
        random_state=42
    )
    
    mlp.fit(X_train, y_train, epochs=1500, verbose=True)
    
    print("\n6. Making predictions...")
    train_predictions = mlp.predict_classes(X_train)
    test_predictions = mlp.predict_classes(X_test)
    
    train_true_labels = np.argmax(y_train, axis=1)
    test_true_labels = np.argmax(y_test, axis=1)
    train_accuracy = np.mean(train_predictions == train_true_labels)
    test_accuracy = np.mean(test_predictions == test_true_labels)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\n7. Plotting results...")
    plot_decision_boundary(mlp, X_normalized, y)
    plot_training_history(mlp.history, "Multi-class Classification Training")


if __name__ == "__main__":
    main()