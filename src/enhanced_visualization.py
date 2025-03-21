"""
Enhanced Data Visualization Module for ML Simulation Environment.

This module provides advanced visualization capabilities for inspecting
data splits, model inputs, outputs, and tracing residuals back to specific images.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf


class DataVisualizer:
    """
    A class for enhanced visualization of data splits, model inputs, outputs,
    and tracing residuals back to specific images.
    
    This class extends the basic visualization capabilities with more detailed
    inspection tools for understanding model behavior and residuals analysis.
    """
    
    def __init__(self, save_dir: str = './results/visualization'):
        """
        Initialize the DataVisualizer.
        
        Args:
            save_dir: Directory to save visualization results
        """
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set default style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def visualize_data_splits(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             n_samples_per_class: int = 2,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize samples from each data split (train, validation, test).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            class_names: Names of the classes
            n_samples_per_class: Number of samples to show per class
            save_path: Path to save the visualization
        """
        # Get number of classes
        n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
        
        # Determine class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Ensure X arrays have the right shape for visualization
        X_train_vis = self._prepare_for_visualization(X_train)
        X_val_vis = self._prepare_for_visualization(X_val)
        X_test_vis = self._prepare_for_visualization(X_test)
        
        # Create figure
        fig = plt.figure(figsize=(15, n_classes * 4))
        
        # Create GridSpec
        gs = GridSpec(n_classes, 3 * n_samples_per_class + 1, figure=fig)
        
        # Add title for each split
        plt.figtext(0.2, 0.95, 'Training Set', fontsize=14, ha='center')
        plt.figtext(0.5, 0.95, 'Validation Set', fontsize=14, ha='center')
        plt.figtext(0.8, 0.95, 'Test Set', fontsize=14, ha='center')
        
        # Plot samples for each class
        for cls_idx in range(n_classes):
            # Get indices for each class in each split
            train_indices = np.where(y_train == cls_idx)[0]
            val_indices = np.where(y_val == cls_idx)[0]
            test_indices = np.where(y_test == cls_idx)[0]
            
            # Add class label
            ax = fig.add_subplot(gs[cls_idx, 0])
            ax.text(0.5, 0.5, class_names[cls_idx], fontsize=12, ha='center', va='center')
            ax.axis('off')
            
            # Plot training samples
            for i in range(min(n_samples_per_class, len(train_indices))):
                idx = train_indices[i]
                ax = fig.add_subplot(gs[cls_idx, i + 1])
                ax.imshow(X_train_vis[idx], cmap='gray')
                ax.set_title(f'Train #{idx}')
                ax.axis('off')
            
            # Plot validation samples
            for i in range(min(n_samples_per_class, len(val_indices))):
                idx = val_indices[i]
                ax = fig.add_subplot(gs[cls_idx, i + n_samples_per_class + 1])
                ax.imshow(X_val_vis[idx], cmap='gray')
                ax.set_title(f'Val #{idx}')
                ax.axis('off')
            
            # Plot test samples
            for i in range(min(n_samples_per_class, len(test_indices))):
                idx = test_indices[i]
                ax = fig.add_subplot(gs[cls_idx, i + 2 * n_samples_per_class + 1])
                ax.imshow(X_test_vis[idx], cmap='gray')
                ax.set_title(f'Test #{idx}')
                ax.axis('off')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('Data Split Visualization', fontsize=16, y=0.98)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_model_predictions(self,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  y_pred: np.ndarray,
                                  softmax_outputs: np.ndarray,
                                  class_names: Optional[List[str]] = None,
                                  n_samples: int = 20,
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize model predictions alongside ground truth labels.
        
        Args:
            X_test: Test features
            y_test: Test labels (ground truth)
            y_pred: Predicted labels
            softmax_outputs: Softmax outputs from the model
            class_names: Names of the classes
            n_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        # Get number of classes
        #n_classes = softmax_outputs.shape[1]
        if class_names is not None:
            n_classes = len(class_names)
        else:
            n_classes = len(np.unique(y_test))
        
        # Determine class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Ensure X_test has the right shape for visualization
        X_test_vis = self._prepare_for_visualization(X_test)
        
        # Limit number of samples
        n_samples = min(n_samples, len(X_test_vis))
        
        # Create figure
        fig = plt.figure(figsize=(15, n_samples * 2.5))
        
        # Create GridSpec
        gs = GridSpec(n_samples, 2 + n_classes, figure=fig)
        
        # Add column headers
        plt.figtext(0.15, 0.95, 'Image', fontsize=14, ha='center')
        plt.figtext(0.35, 0.95, 'Prediction', fontsize=14, ha='center')
        plt.figtext(0.7, 0.95, 'Class Probabilities', fontsize=14, ha='center')
        
        # Plot samples
        for i in range(n_samples):
            # Plot image
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(X_test_vis[i], cmap='gray')
            ax.set_title(f'True: {class_names[y_test[i]]}')
            ax.axis('off')
            
            # Plot prediction info
            ax = fig.add_subplot(gs[i, 1])
            ax.text(0.5, 0.5, f'Pred: {class_names[y_pred[i]]}', fontsize=12, ha='center', va='center')
            if y_test[i] == y_pred[i]:
                ax.set_facecolor('lightgreen')
            else:
                ax.set_facecolor('lightcoral')
            ax.axis('off')
            
            # Plot class probabilities
            for j in range(n_classes):
                ax = fig.add_subplot(gs[i, j + 2])
                ax.barh([0], [softmax_outputs[i, j]], color='skyblue')
                ax.set_xlim(0, 1)
                ax.set_title(f'{class_names[j]}: {softmax_outputs[i, j]:.2f}')
                ax.set_yticks([])
                if j == y_test[i]:
                    ax.spines['top'].set_color('green')
                    ax.spines['bottom'].set_color('green')
                    ax.spines['left'].set_color('green')
                    ax.spines['right'].set_color('green')
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('Model Predictions Visualization', fontsize=16, y=0.98)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_confusion_matrix_with_examples(self, prepared_confusion_matrix, class_names, examples, save_path):
        """Visualize confusion matrix with example images."""
        fig = plt.figure(figsize=(10, 10))
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(prepared_confusion_matrix))]
        
        grid_size = len(class_names)

        
        for k in range(grid_size * grid_size):
            mini_ax = plt.subplot2grid((grid_size, grid_size), (k // grid_size, k % grid_size))
            mini_ax.axis('off')
            
            if k < len(examples):
                mini_ax.imshow(examples[k], cmap='gray')
                mini_ax.set_title(f"{class_names[k // grid_size]} vs {class_names[k % grid_size]}")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    
    def visualize_residuals_by_image(self,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   y_pred: np.ndarray,
                                   intermediate_reps: np.ndarray,
                                   softmax_outputs: np.ndarray,
                                   similarity_matrices: Dict[int, np.ndarray],
                                   entropies: np.ndarray,
                                   class_names: Optional[List[str]] = None,
                                   n_samples: int = 10,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize residuals analysis traced back to specific images.
        
        Args:
            X_test: Test features
            y_test: Test labels (ground truth)
            y_pred: Predicted labels
            intermediate_reps: Intermediate representations from the model
            softmax_outputs: Softmax outputs from the model
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            entropies: Entropy values for each observation
            class_names: Names of the classes
            n_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        # Get number of classes
        n_classes = softmax_outputs.shape[1]
        
        # Determine class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Ensure X_test has the right shape for visualization
        X_test_vis = self._prepare_for_visualization(X_test)
        
        # Limit number of samples
        n_samples = min(n_samples, len(X_test_vis))
        
        # Calculate average similarity for each class
        average_similarities = {}
        for cls, similarity_matrix in similarity_matrices.items():
            indices = np.where(y_pred == cls)[0]
            if len(indices) <= 1:
                continue
                
            n_samples_cls = similarity_matrix.shape[0]
            avg_similarity = np.zeros(n_samples_cls)
            
            for i in range(n_samples_cls):
                similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i+1:]
                ])
                avg_similarity[i] = np.mean(similarities)
            
            # Map back to original indices
            for i, idx in enumerate(indices):
                if idx < len(X_test_vis):
                    average_similarities[idx] = avg_similarity[i]
        
        # Create figure
        fig = plt.figure(figsize=(15, n_samples * 3))
        
        # Create GridSpec
        gs = GridSpec(n_samples, 4, figure=fig, width_ratios=[1, 1, 2, 2])
        
        # Add column headers
        plt.figtext(0.125, 0.95, 'Image', fontsize=14, ha='center')
        plt.figtext(0.375, 0.95, 'Prediction', fontsize=14, ha='center')
        plt.figtext(0.625, 0.95, 'Entropy & Similarity', fontsize=14, ha='center')
        plt.figtext(0.875, 0.95, 'Class Probabilities', fontsize=14, ha='center')
        
        # Sort samples by entropy for more informative visualization
        sorted_indices = np.argsort(entropies)[::-1]  # High entropy first
        sorted_indices = sorted_indices[:n_samples]
        
        # Plot samples
        for i, idx in enumerate(sorted_indices):
            # Plot image
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(X_test_vis[idx], cmap='gray')
            ax.set_title(f'True: {class_names[y_test[idx]]}')
            ax.axis('off')
            
            # Plot prediction info
            ax = fig.add_subplot(gs[i, 1])
            ax.text(0.5, 0.5, f'Pred: {class_names[y_pred[idx]]}', fontsize=12, ha='center', va='center')
            if y_test[idx] == y_pred[idx]:
                ax.set_facecolor('lightgreen')
            else:
                ax.set_facecolor('lightcoral')
            ax.axis('off')
            
            # Plot entropy and similarity
            ax = fig.add_subplot(gs[i, 2])
            metrics = [
                ('Entropy', entropies[idx]),
                ('Avg Similarity', average_similarities.get(idx, np.nan))
            ]
            
            y_pos = np.arange(len(metrics))
            values = [m[1] for m in metrics]
            
            ax.barh(y_pos, values, color=['skyblue', 'lightgreen'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([m[0] for m in metrics])
            ax.set_xlim(0, max(1.0, max(filter(lambda x: not np.isnan(x), values)) * 1.1))
            ax.set_title('Residuals Metrics')
            
            # Plot class probabilities
            ax = fig.add_subplot(gs[i, 3])
            y_pos = np.arange(n_classes)
            ax.barh(y_pos, softmax_outputs[idx], color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names)
            ax.set_xlim(0, 1)
            ax.set_title('Class Probabilities')
            
            # Highlight true class
            true_class_idx = np.where(y_pos == y_test[idx])[0][0]
            ax.get_children()[true_class_idx].set_color('green')
            
            # Highlight predicted class if different
            if y_test[idx] != y_pred[idx]:
                pred_class_idx = np.where(y_pos == y_pred[idx])[0][0]
                ax.get_children()[pred_class_idx].set_color('red')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('Residuals Analysis by Image', fontsize=16, y=0.98)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_intermediate_representations(self,
                                            X_test: np.ndarray,
                                            y_test: np.ndarray,
                                            y_pred: np.ndarray,
                                            intermediate_reps: np.ndarray,
                                            class_names: Optional[List[str]] = None,
                                            n_samples: int = 10,
                                            save_path: Optional[str] = None) -> None:
        """
        Visualize intermediate representations alongside original images.
        
        Args:
            X_test: Test features
            y_test: Test labels (ground truth)
            y_pred: Predicted labels
            intermediate_reps: Intermediate representations from the model
            class_names: Names of the classes
            n_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        # Get number of classes
        n_classes = len(np.unique(np.concatenate([y_test, y_pred])))
        
        # Determine class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Ensure X_test has the right shape for visualization
        X_test_vis = self._prepare_for_visualization(X_test)
        
        # Limit number of samples
        n_samples = min(n_samples, len(X_test_vis))
        
        # Create figure
        fig = plt.figure(figsize=(15, n_samples * 2.5))
        
        # Create GridSpec
        gs = GridSpec(n_samples, 3, figure=fig, width_ratios=[1, 1, 3])
        
        # Add column headers
        plt.figtext(0.15, 0.95, 'Image', fontsize=14, ha='center')
        plt.figtext(0.4, 0.95, 'Prediction', fontsize=14, ha='center')
        plt.figtext(0.75, 0.95, 'Intermediate Representation', fontsize=14, ha='center')
        
        # Sort samples to show a mix of correct and incorrect predictions
        correct_indices = np.where(y_test == y_pred)[0]
        incorrect_indices = np.where(y_test != y_pred)[0]
        
        # Prioritize showing some of each
        n_correct = min(n_samples // 2, len(correct_indices))
        n_incorrect = min(n_samples - n_correct, len(incorrect_indices))
        
        # Adjust n_correct if we don't have enough incorrect samples
        n_correct = n_samples - n_incorrect
        
        # Select samples
        selected_indices = np.concatenate([
            np.random.choice(correct_indices, n_correct, replace=False),
            np.random.choice(incorrect_indices, n_incorrect, replace=False)
        ])
        
        # Plot samples
        for i, idx in enumerate(selected_indices[:n_samples]):
            # Plot image
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(X_test_vis[idx], cmap='gray')
            ax.set_title(f'True: {class_names[y_test[idx]]}')
            ax.axis('off')
            
            # Plot prediction info
            ax = fig.add_subplot(gs[i, 1])
            ax.text(0.5, 0.5, f'Pred: {class_names[y_pred[idx]]}', fontsize=12, ha='center', va='center')
            if y_test[idx] == y_pred[idx]:
                ax.set_facecolor('lightgreen')
            else:
                ax.set_facecolor('lightcoral')
            ax.axis('off')
            
            # Plot intermediate representation
            ax = fig.add_subplot(gs[i, 2])
            rep = intermediate_reps[idx]
            ax.bar(range(len(rep)), rep, color='skyblue')
            ax.set_title(f'Representation (dim={len(rep)})')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Activation')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('Intermediate Representations Visualization', fontsize=16, y=0.98)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_similarity_matrix_with_images(self,
                                             X_test: np.ndarray,
                                             y_pred: np.ndarray,
                                             similarity_matrices: Dict[int, np.ndarray],
                                             class_names: Optional[List[str]] = None,
                                             class_to_show: Optional[int] = None,
                                             save_path: Optional[str] = None) -> None:
        """
        Visualize similarity matrix with corresponding images for a specific class.
        
        Args:
            X_test: Test features
            y_pred: Predicted labels
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            class_names: Names of the classes
            class_to_show: Class to visualize (if None, will use the class with most samples)
            save_path: Path to save the visualization
        """
        # Get number of classes
        n_classes = len(np.unique(y_pred))
        
        # Determine class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Ensure X_test has the right shape for visualization
        X_test_vis = self._prepare_for_visualization(X_test)
        
        # If class_to_show is not specified, use the class with most samples
        if class_to_show is None:
            class_counts = {cls: len(np.where(y_pred == cls)[0]) for cls in range(n_classes)}
            class_to_show = max(class_counts, key=class_counts.get)
        
        # Get indices for the selected class
        class_indices = np.where(y_pred == class_to_show)[0]
        
        # Check if we have similarity data for this class
        if class_to_show not in similarity_matrices or len(class_indices) <= 1:
            print(f"No similarity data available for class {class_to_show}")
            return
        
        # Get similarity matrix for this class
        similarity_matrix = similarity_matrices[class_to_show]
        
        # Create figure
        n_samples = len(class_indices)
        fig_size = max(8, min(20, n_samples + 2))  # Scale figure size with number of samples
        fig = plt.figure(figsize=(fig_size, fig_size))
        
        # Create GridSpec
        gs = GridSpec(n_samples + 1, n_samples + 1, figure=fig)
        
        # Plot similarity matrix
        ax_matrix = fig.add_subplot(gs[1:, 1:])
        im = ax_matrix.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax_matrix.set_title(f'Similarity Matrix for {class_names[class_to_show]}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax_matrix)
        
        # Plot images along the top and left
        for i, idx in enumerate(class_indices):
            # Plot image on top
            ax_top = fig.add_subplot(gs[0, i + 1])
            ax_top.imshow(X_test_vis[idx], cmap='gray')
            ax_top.set_title(f'#{idx}')
            ax_top.axis('off')
            
            # Plot image on left
            ax_left = fig.add_subplot(gs[i + 1, 0])
            ax_left.imshow(X_test_vis[idx], cmap='gray')
            ax_left.set_title(f'#{idx}')
            ax_left.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        plt.suptitle(f'Similarity Matrix with Images for {class_names[class_to_show]}', 
                    fontsize=16, y=1.02)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _prepare_for_visualization(self, images: np.ndarray) -> np.ndarray:
        """
        Prepare images for visualization by ensuring they have the right shape.
        
        Args:
            images: Image array
            
        Returns:
            Processed images ready for visualization
        """
        # Make a copy to avoid modifying the original
        images_vis = images.copy()
        
        # If images have a channel dimension, remove it for visualization
        if len(images_vis.shape) == 4:  # (samples, height, width, channels)
            images_vis = images_vis.squeeze(axis=3)
        
        return images_vis
    
    def run_comprehensive_visualization(self,
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_val: np.ndarray,
                                      y_val: np.ndarray,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      y_pred: np.ndarray,
                                      intermediate_reps: np.ndarray,
                                      softmax_outputs: np.ndarray,
                                      similarity_matrices: Dict[int, np.ndarray],
                                      entropies: np.ndarray,
                                      class_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Run a comprehensive visualization of all aspects of the model and data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels (ground truth)
            y_pred: Predicted labels
            intermediate_reps: Intermediate representations from the model
            softmax_outputs: Softmax outputs from the model
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            entropies: Entropy values for each observation
            class_names: Names of the classes
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_test)))]        # we could also get the class names from the size of similarity_matrices

        # Create visualization directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Dictionary to store visualization paths
        visualization_paths = {}
        
        # 1. Visualize data splits
        data_splits_path = os.path.join(self.save_dir, 'data_splits.png')
        self.visualize_data_splits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            save_path=data_splits_path
        )
        visualization_paths['data_splits'] = data_splits_path
        
        # 2. Visualize model predictions
        predictions_path = os.path.join(self.save_dir, 'model_predictions.png')
        self.visualize_model_predictions(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            softmax_outputs=softmax_outputs,
            class_names=class_names,
            save_path=predictions_path
        )
        visualization_paths['model_predictions'] = predictions_path
        
        # 3. Visualize confusion matrix with examples
        confusion_matrix_path = os.path.join(self.save_dir, 'confusion_matrix_with_examples.png')
        prepared_confusion_matrix = confusion_matrix(y_test, y_pred)
        examples = X_test[:len(prepared_confusion_matrix)]  # Assuming examples are the first few test samples
        self.visualize_confusion_matrix_with_examples(
            prepared_confusion_matrix=prepared_confusion_matrix,
            class_names=class_names,
            examples=examples,
            save_path=confusion_matrix_path
        )
        visualization_paths['confusion_matrix'] = confusion_matrix_path
        
        # 4. Visualize residuals by image
        residuals_path = os.path.join(self.save_dir, 'residuals_by_image.png')
        self.visualize_residuals_by_image(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            intermediate_reps=intermediate_reps,
            softmax_outputs=softmax_outputs,
            similarity_matrices=similarity_matrices,
            entropies=entropies,
            class_names=class_names,
            save_path=residuals_path
        )
        visualization_paths['residuals'] = residuals_path
        
        # 5. Visualize intermediate representations
        representations_path = os.path.join(self.save_dir, 'intermediate_representations.png')
        self.visualize_intermediate_representations(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            intermediate_reps=intermediate_reps,
            class_names=class_names,
            save_path=representations_path
        )
        visualization_paths['intermediate_representations'] = representations_path
        
        # 6. Visualize similarity matrix with images for each class
        for cls in similarity_matrices.keys():
            similarity_path = os.path.join(self.save_dir, f'similarity_matrix_class_{cls}.png')
            self.visualize_similarity_matrix_with_images(
                X_test=X_test,
                y_pred=y_pred,
                similarity_matrices=similarity_matrices,
                class_names=class_names,
                class_to_show=cls,
                save_path=similarity_path
            )
            visualization_paths[f'similarity_matrix_class_{cls}'] = similarity_path
        
        print(f"Comprehensive visualization complete. Results saved to: {self.save_dir}")
        return visualization_paths
