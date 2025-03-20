"""
Model Training Utilities for ML Simulation Environment.

This module provides functionality for resuming training of previously saved models
and other training-related utilities.
"""

import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import json


class TrainingUtilities:
    """
    A class for model training utilities, including resuming training of previously saved models.
    
    This class provides functionality for loading saved models, resuming training with
    additional epochs, and managing training history.
    """
    
    def __init__(self, save_dir: str = './results'):
        """
        Initialize the TrainingUtilities.
        
        Args:
            save_dir: Directory where models and training history are saved
        """
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def resume_training(self,
                       model_path: str,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None,
                       additional_epochs: int = 10,
                       batch_size: int = 32,
                       callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
                       verbose: int = 1,
                       save_best_only: bool = True,
                       **kwargs) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
        """
        Resume training of a previously saved Keras model.
        
        Args:
            model_path: Path to the saved model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            additional_epochs: Number of additional epochs to train
            batch_size: Number of samples per gradient update
            callbacks: List of callbacks to apply during training
            verbose: Verbosity mode
            save_best_only: Whether to save only the best model during training
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (trained model, combined training history)
        """
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load previous training history if available
        history_path = os.path.join(os.path.dirname(model_path), 'training_history.json')
        previous_history = {}
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    previous_history = json.load(f)
                print(f"Loaded training history from {history_path}")
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")
        
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
        
        # Prepare callbacks
        if callbacks is None:
            callbacks = []
        
        # Add ModelCheckpoint callback if save_best_only is True
        if save_best_only:
            checkpoint_path = os.path.join(os.path.dirname(model_path), 'best_model.h5')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if validation_data is not None else 'loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Resume training
        print(f"Resuming training for {additional_epochs} additional epochs...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=additional_epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        # Combine previous history with new history
        combined_history = {}
        
        # Initialize combined history with previous history
        for key, values in previous_history.items():
            combined_history[key] = values.copy()
        
        # Add new history
        for key, values in history.history.items():
            if key in combined_history:
                combined_history[key].extend(values)
            else:
                combined_history[key] = values
        
        # Save combined history
        with open(history_path, 'w') as f:
            json.dump(combined_history, f)
        
        # Save the final model
        final_model_path = os.path.join(os.path.dirname(model_path), 'final_model.h5')
        model.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        return model, combined_history
    
    def plot_training_history(self,
                            history: Dict[str, List[float]],
                            metrics: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history metrics.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot (if None, will plot all metrics)
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Determine metrics to plot
        if metrics is None:
            metrics = [key for key in history.keys() if not key.startswith('val_')]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            plt.subplot(len(metrics), 1, i + 1)
            
            # Plot training metric
            plt.plot(history[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                plt.plot(history[val_metric], label=f'Validation {metric}')
            
            plt.title(f'{metric.capitalize()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def find_saved_models(self, base_dir: Optional[str] = None) -> List[str]:
        """
        Find all saved models in the specified directory.
        
        Args:
            base_dir: Base directory to search (if None, uses self.save_dir)
            
        Returns:
            List of paths to saved models
        """
        if base_dir is None:
            base_dir = self.save_dir
        
        model_paths = []
        
        # Walk through the directory
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.h5'):
                    model_paths.append(os.path.join(root, file))
        
        return model_paths
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary containing model information
        """
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Get model information
        info = {
            'path': model_path,
            'name': os.path.basename(model_path),
            'layers': len(model.layers),
            'parameters': model.count_params(),
            'input_shape': model.input_shape[1:],
            'output_shape': model.output_shape[1:],
        }
        
        # Load training history if available
        history_path = os.path.join(os.path.dirname(model_path), 'training_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # Add history information
                info['epochs'] = len(history.get('loss', []))
                
                # Add final metrics
                for key, values in history.items():
                    if len(values) > 0:
                        info[f'final_{key}'] = values[-1]
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")
        
        return info
    
    def compare_models(self, 
                     model_paths: List[str],
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     figsize: Tuple[int, int] = (12, 8),
                     save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple saved models on test data.
        
        Args:
            model_paths: List of paths to saved models
            X_test: Test features
            y_test: Test labels
            figsize: Figure size
            save_path: Path to save the comparison plot
            
        Returns:
            Dictionary mapping model names to evaluation metrics
        """
        # Ensure X_test is in the correct format for Keras
        if len(X_test.shape) == 3:  # If it's a 3D array (samples, height, width)
            X_test = X_test[..., np.newaxis]  # Add channel dimension
        
        # Dictionary to store evaluation results
        results = {}
        
        # Evaluate each model
        for model_path in model_paths:
            # Load the model
            model = tf.keras.models.load_model(model_path)
            model_name = os.path.basename(model_path)
            
            # Evaluate the model
            evaluation = model.evaluate(X_test, y_test, verbose=0)
            
            # Get metric names
            metric_names = model.metrics_names
            
            # Store evaluation metrics
            metrics = {metric_names[i]: evaluation[i] for i in range(len(metric_names))}
            
            # Get predictions
            y_pred = np.argmax(model.predict(X_test), axis=1)
            
            # Calculate additional metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision_micro'] = precision_score(y_test, y_pred, average='micro', zero_division=0)
            metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
            metrics['recall_micro'] = recall_score(y_test, y_pred, average='micro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
            metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Store results
            results[model_name] = metrics
        
        # Create comparison plot
        plt.figure(figsize=figsize)
        
        # Get common metrics across all models
        common_metrics = set.intersection(*[set(metrics.keys()) for metrics in results.values()])
        
        # Filter out non-numeric metrics
        numeric_metrics = [metric for metric in common_metrics 
                         if all(isinstance(results[model][metric], (int, float)) 
                              for model in results.keys())]
        
        # Sort metrics by name
        numeric_metrics = sorted(numeric_metrics)
        
        # Create bar chart for each metric
        for i, metric in enumerate(numeric_metrics):
            plt.subplot(len(numeric_metrics), 1, i + 1)
            
            # Get values for this metric
            models = list(results.keys())
            values = [results[model][metric] for model in models]
            
            # Create bar chart
            bars = plt.bar(models, values, color='skyblue')
            
            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', rotation=0)
            
            plt.title(f'{metric.capitalize()} Comparison')
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return results
