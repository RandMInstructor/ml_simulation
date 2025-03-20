"""
Training and Evaluation Pipeline for ML Simulation Environment.

This module provides functionality for training, evaluating, and saving
machine learning models for multi-class classification problems.
"""

import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any, Optional, List, Union, Tuple, Callable


class TrainingPipeline:
    """
    A class for training and evaluating machine learning models.
    
    This class provides functionality for training, evaluating, and saving
    machine learning models for multi-class classification problems.
    """
    
    def __init__(self, model: Any, model_type: str, save_dir: str = './results'):
        """
        Initialize the TrainingPipeline.
        
        Args:
            model: The machine learning model to train and evaluate
            model_type: Type of the model ('keras' or 'sklearn')
            save_dir: Directory to save model and results
        """
        self.model = model
        self.model_type = model_type.lower()
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Dictionary to store training history
        self.history = {}
        
        # Dictionary to store evaluation metrics
        self.metrics = {}
        
        # Dictionary to store intermediate representations
        self.intermediate_representations = {}
        
        # Dictionary to store softmax outputs
        self.softmax_outputs = {}
        
        # Validate model type
        if self.model_type not in ['keras', 'sklearn']:
            raise ValueError(f"Unknown model type: {model_type}. Must be 'keras' or 'sklearn'.")
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        if self.model_type == 'keras':
            return self._train_keras(X_train, y_train, X_val, y_val, **kwargs)
        else:  # sklearn
            return self._train_sklearn(X_train, y_train, **kwargs)
    
    def _train_keras(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    batch_size: int = 32,
                    epochs: int = 100,
                    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
                    verbose: int = 1,
                    **kwargs) -> Dict[str, Any]:
        """
        Train a Keras model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
            callbacks: List of callbacks to apply during training
            verbose: Verbosity mode
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        # Ensure X_train is in the correct format for Keras
        if len(X_train.shape) == 3:  # If it's a 3D array (samples, height, width)
            X_train = X_train[..., np.newaxis]  # Add channel dimension
        
        # Ensure X_val is in the correct format for Keras if provided
        if X_val is not None and len(X_val.shape) == 3:
            X_val = X_val[..., np.newaxis]  # Add channel dimension
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        # Store training history
        self.history = history.history
        
        return self.history
    
    def _train_sklearn(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      **kwargs) -> Dict[str, Any]:
        """
        Train a scikit-learn model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional training parameters
            
        Returns:
            Training history (empty for scikit-learn models)
        """
        # Ensure X_train is in the correct format for sklearn
        if len(X_train.shape) > 2:  # If it's a multi-dimensional array
            # Flatten all dimensions except the first (samples)
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        # Train the model
        self.model.fit(X_train, y_train, **kwargs)
        
        # scikit-learn models don't have training history
        self.history = {}
        
        return self.history
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray,
                **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        if self.model_type == 'keras':
            return self._evaluate_keras(X_test, y_test, **kwargs)
        else:  # sklearn
            return self._evaluate_sklearn(X_test, y_test, **kwargs)
    
    def _evaluate_keras(self, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       batch_size: int = 32,
                       verbose: int = 1,
                       **kwargs) -> Dict[str, float]:
        """
        Evaluate a Keras model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Number of samples per batch
            verbose: Verbosity mode
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        # Ensure X_test is in the correct format for Keras
        if len(X_test.shape) == 3:  # If it's a 3D array (samples, height, width)
            X_test = X_test[..., np.newaxis]  # Add channel dimension
        
        # Evaluate the model
        evaluation = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose, **kwargs)
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        # Store evaluation metrics
        self.metrics = {metric_names[i]: evaluation[i] for i in range(len(metric_names))}
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate additional metrics
        self.metrics.update(self._calculate_additional_metrics(y_test, y_pred))
        
        return self.metrics
    
    def _evaluate_sklearn(self, 
                         X_test: np.ndarray, 
                         y_test: np.ndarray,
                         **kwargs) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        # Ensure X_test is in the correct format for sklearn
        if len(X_test.shape) > 2:  # If it's a multi-dimensional array
            # Flatten all dimensions except the first (samples)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store evaluation metrics
        self.metrics = {'accuracy': accuracy}
        
        # Calculate additional metrics
        self.metrics.update(self._calculate_additional_metrics(y_test, y_pred))
        
        return self.metrics
    
    def _calculate_additional_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Additional metrics
        """
        # Calculate precision, recall, and F1 score
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Return additional metrics
        return {
            'precision_micro': precision_micro,
            'precision_macro': precision_macro,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted labels
        """
        if self.model_type == 'keras':
            # Ensure X is in the correct format for Keras
            if len(X.shape) == 3:  # If it's a 3D array (samples, height, width)
                X = X[..., np.newaxis]  # Add channel dimension
            
            # Get raw predictions (probabilities)
            y_prob = self.model.predict(X, **kwargs)
            # Convert to class labels
            y_pred = np.argmax(y_prob, axis=1)
        else:  # sklearn
            # Ensure X is in the correct format for sklearn
            if len(X.shape) > 2:  # If it's a multi-dimensional array
                # Flatten all dimensions except the first (samples)
                X = X.reshape(X.shape[0], -1)
            
            # Get class labels directly
            y_pred = self.model.predict(X, **kwargs)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Probability estimates
        """
        if self.model_type == 'keras':
            # Ensure X is in the correct format for Keras
            if len(X.shape) == 3:  # If it's a 3D array (samples, height, width)
                X = X[..., np.newaxis]  # Add channel dimension
            
            # Get raw predictions (probabilities)
            y_prob = self.model.predict(X, **kwargs)
        else:  # sklearn
            # Ensure X is in the correct format for sklearn
            if len(X.shape) > 2:  # If it's a multi-dimensional array
                # Flatten all dimensions except the first (samples)
                X = X.reshape(X.shape[0], -1)
            
            # Get probability estimates
            y_prob = self.model.predict_proba(X, **kwargs)
        
        return y_prob
    
    def extract_intermediate_representations(self, 
                                           X: np.ndarray, 
                                           intermediate_model: tf.keras.Model,
                                           **kwargs) -> np.ndarray:
        """
        Extract intermediate representations from the model.
        
        Args:
            X: Input features
            intermediate_model: Model to extract intermediate representations
            **kwargs: Additional prediction parameters
            
        Returns:
            Intermediate representations
        """
        # Ensure X is in the correct format for Keras
        if len(X.shape) == 3:  # If it's a 3D array (samples, height, width)
            X = X[..., np.newaxis]  # Add channel dimension
        
        # Extract intermediate representations
        representations = intermediate_model.predict(X, **kwargs)
        
        # Store intermediate representations
        self.intermediate_representations = representations
        
        return representations
    
    def extract_softmax_outputs(self, 
                              X: np.ndarray, 
                              softmax_model: tf.keras.Model,
                              **kwargs) -> np.ndarray:
        """
        Extract softmax outputs from the model.
        
        Args:
            X: Input features
            softmax_model: Model to extract softmax outputs
            **kwargs: Additional prediction parameters
            
        Returns:
            Softmax outputs
        """
        # Ensure X is in the correct format for Keras
        if len(X.shape) == 3:  # If it's a 3D array (samples, height, width)
            X = X[..., np.newaxis]  # Add channel dimension
        
        # Extract softmax outputs
        outputs = softmax_model.predict(X, **kwargs)
        
        # Store softmax outputs
        self.softmax_outputs = outputs
        
        return outputs
    
    def plot_training_history(self, 
                            metrics: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 6),
                            save_path: Optional[str] = None) -> None:
        """
        Plot the training history.
        
        Args:
            metrics: List of metrics to plot (default: ['loss', 'accuracy'])
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.history:
            print("No training history available.")
            return
        
        # Default metrics to plot
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        # Filter metrics that are in the history
        metrics = [m for m in metrics if m in self.history]
        
        if not metrics:
            print("No valid metrics to plot.")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i + 1)
            
            # Plot training metric
            plt.plot(self.history[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in self.history:
                plt.plot(self.history[val_metric], label=f'Validation {metric}')
            
            # Add labels and legend
            plt.title(f'{metric.capitalize()} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            # Check if the save directory exists - ensure it does
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, 
                            X_test: np.ndarray, 
                            y_test: np.ndarray,
                            normalize: Optional[str] = None,
                            class_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 8),
                            cmap: str = 'Blues',
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Plot the confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
            class_names: Names of the classes
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save the plot
            
        Returns:
            Confusion matrix
        """
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix if requested
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            title = 'Normalized Confusion Matrix (True)'
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
            title = 'Normalized Confusion Matrix (Pred)'
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
            title = 'Normalized Confusion Matrix (All)'
        else:
            title = 'Confusion Matrix'
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
        plt.title(title, fontsize=14)
        plt.colorbar()
        
        # Add class labels
        n_classes = cm.shape[0]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Add labels
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return cm
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model (default: {save_dir}/model)
            
        Returns:
            Path where the model was saved
        """
        if path is None:
            path = os.path.join(self.save_dir, 'model')
        
        if self.model_type == 'keras':
            # Save Keras model
            self.model.save(f"{path}.h5")
            saved_path = f"{path}.h5"
        else:  # sklearn
            # Save scikit-learn model
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            saved_path = f"{path}.pkl"
        
        return saved_path
    
    def load_model(self, path: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        if path.endswith('.h5'):
            # Load Keras model
            self.model = tf.keras.models.load_model(path)
            self.model_type = 'keras'
        elif path.endswith('.pkl'):
            # Load scikit-learn model
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'sklearn'
        else:
            raise ValueError(f"Unknown model format: {path}")
        
        return self.model

# Write a function to split a training set into a training and validation set
def train_val_split(X_train: np.ndarray, y_train: np.ndarray, val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a training set into a training and validation set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        val_size: Fraction of the training data to use as validation
        
    Returns:
        Training features, validation features, training labels, validation labels
    """
    # Determine the size of the validation set
    val_size = int(len(X_train) * val_size)
    
    # Split the training data into training and validation sets
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    return (X_train, y_train) , (X_val, y_val)


# Simple Model
def get_simple_example_model() -> tf.keras.Model:
        # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
    

# Basic MNIST Example
def get_basic_mnist_example_model() -> tf.keras.Model:
    # Define the model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def basic_public_mnist_example():
    """
    Basic example using the MNIST dataset.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Create a separate validation set
    x_train, x_val = x_train[:-5000], x_train[-5000:]
    y_train, y_val = y_train[:-5000], y_train[-5000:]

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define the model
    model = get_basic_mnist_example_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

def demo_incorporated_mnist_example(model, save_dir = './results/training_pipeline_main'):
    """
    Basic example using the MNIST dataset to test all custom functions.
    """

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Create a separate validation set
    (x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train, val_size=0.1)

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0


    #model = mnist_model_selection()
    #model = get_basic_mnist_model()
    #model = get_model_selector_cnn(input_shape=(28, 28, 1))

    # Create a training pipeline
    pipeline = TrainingPipeline(model=model, model_type='keras')
    
    # Train the model
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train with validation
    history = pipeline.train(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        batch_size=32,
        epochs=5,
        callbacks=callbacks
    )
    
    # Evaluate the model
    metrics = pipeline.evaluate(X_test=x_test, y_test=y_test)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history
    pipeline.plot_training_history(save_path=f"{save_dir}/training_history.png")
    
    # Plot confusion matrix
    pipeline.plot_confusion_matrix(X_test=x_test, y_test=y_test)

def original_main(model):
    
    # Establish save path
    save_dir = './results/training_pipeline_main'


    
    # Create a training pipeline
    pipeline = TrainingPipeline(model=model, model_type='keras')
    
    # Generate some dummy data
    X_train = np.random.randn(100, 32, 32, 1)
    y_train = np.random.randint(0, 10, size=100)
    X_test = np.random.randn(20, 32, 32, 1)
    y_test = np.random.randint(0, 10, size=20)
    
    # Train the model
    history = pipeline.train(
        X_train=X_train,
        y_train=y_train,
        batch_size=32,
        epochs=5
    )
    
    # Evaluate the model
    metrics = pipeline.evaluate(X_test=X_test, y_test=y_test)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history
    pipeline.plot_training_history(save_path=f"{save_dir}/training_history.png")
    
    # Plot confusion matrix
    pipeline.plot_confusion_matrix(X_test=X_test, y_test=y_test)

# Example usage
if __name__ == "__main__":
    model = get_simple_example_model()
    original_main(model)
    
    basic_public_mnist_example()

    model = get_basic_mnist_example_model()
    demo_incorporated_mnist_example(model)


