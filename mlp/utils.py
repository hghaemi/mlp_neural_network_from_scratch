import numpy as np
from sklearn.datasets import make_classification, make_regression, make_circles, make_moons
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def generate_classification_data(n_samples=1000, n_features=2, n_classes=2, 
                                n_clusters_per_class=1, random_state=42):

    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
                              n_redundant=0, n_informative=n_features,
                              random_state=random_state)
    return X, y


def generate_regression_data(n_samples=1000, n_features=1, noise=0.1, random_state=42):

    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=random_state)
    return X, y


def generate_circles_data(n_samples=1000, noise=0.1, factor=0.8, random_state=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y


def generate_moons_data(n_samples=1000, noise=0.1, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def generate_spiral_data(n_samples=1000, n_classes=3, random_state=42):

    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype='uint8')
    
    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(j * 4, (j + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    
    return X, y


def normalize_data(X, method='standard'):

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler


def one_hot_encode(y, n_classes=None):

    if n_classes is None:
        n_classes = len(np.unique(y))
    
    one_hot = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    
    return one_hot


def train_test_validation_split(X, y, test_size=0.2, val_size=0.2, random_state=42):

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_decision_boundary(model, X, y, resolution=0.02):

    try:
        import matplotlib.pyplot as plt
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_classes(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Cannot plot decision boundary.")


def plot_training_history(history, title="Training History"):

    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history['loss'])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        if 'accuracy' in history and history['accuracy']:
            axes[1].plot(history['accuracy'])
            axes[1].set_title('Training Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].grid(True)
        else:
            axes[1].set_title('No Accuracy Data')
            axes[1].text(0.5, 0.5, 'Accuracy not tracked\n(likely regression task)', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Cannot plot training history.")