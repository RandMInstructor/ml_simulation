"""
Visualization and Statistical Tests Module for ML Simulation Environment.

This module provides advanced visualization and statistical analysis tools
for the ML simulation environment, focusing on image classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy import stats
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class VisualizationAnalyzer:
    """
    A class for advanced visualization and statistical analysis of model performance
    and residuals in multi-class classification problems.
    
    This class extends the basic visualization capabilities of the ResidualsAnalyzer
    with more advanced plots and statistical tests.
    """
    
    def __init__(self, save_dir: str = './results'):
        """
        Initialize the VisualizationAnalyzer.
        
        Args:
            save_dir: Directory to save visualization results
        """
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Dictionary to store analysis results
        self.results = {}
        
        # Set default style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def plot_correlation_heatmap(self, 
                               similarity_matrices: Dict[int, np.ndarray],
                               entropies: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               title: str = 'Correlation Heatmap',
                               save_path: Optional[str] = None) -> None:
        """
        Plot a heatmap of correlations between similarity and entropy for each class.
        
        Args:
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            entropies: Entropy values for each observation
            y_pred: Predicted class labels
            class_names: Names of the classes
            title: Plot title
            save_path: Path to save the plot
        """
        # Get unique predicted classes
        unique_classes = np.unique(y_pred)
        
        # Determine class names if not provided
        if class_names is None:
            class_names = {cls: f'Class {cls}' for cls in unique_classes}
        elif isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        
        # Calculate average similarity for each class
        average_similarities = {}
        for cls, similarity_matrix in similarity_matrices.items():
            n_samples = similarity_matrix.shape[0]
            avg_similarity = np.zeros(n_samples)
            
            for i in range(n_samples):
                similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i+1:]
                ])
                avg_similarity[i] = np.mean(similarities)
            
            average_similarities[cls] = avg_similarity
        
        # Calculate correlation for each class
        correlation_data = []
        
        for cls in unique_classes:
            # Get indices of observations predicted as this class
            indices = np.where(y_pred == cls)[0]
            
            if len(indices) <= 1 or cls not in average_similarities:
                # Skip classes with only one observation or no similarity data
                continue
            
            # Get average similarity and entropy for this class
            avg_similarity = average_similarities[cls]
            cls_entropies = entropies[indices]
            
            # Calculate Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(avg_similarity, cls_entropies)
            #https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

            # Calculate Spearman's rank correlation
            spearman_corr, spearman_p = stats.spearmanr(avg_similarity, cls_entropies)
            #https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

            # Store correlation data
            correlation_data.append({
                'Class': class_names.get(cls, f'Class {cls}'),
                'Pearson Correlation': pearson_corr,
                'Pearson P-Value': pearson_p,
                'Spearman Correlation': spearman_corr,
                'Spearman P-Value': spearman_p,
                'Sample Size': len(indices)
            })
        
        # Create DataFrame
        df = pd.DataFrame(correlation_data)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot heatmap for Pearson correlation
        plt.subplot(1, 2, 1)
        sns.heatmap(
            df.pivot_table(index='Class', values='Pearson Correlation'),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=.5
        )
        plt.title('Pearson Correlation')
        
        # Plot heatmap for Spearman correlation
        plt.subplot(1, 2, 2)
        sns.heatmap(
            df.pivot_table(index='Class', values='Spearman Correlation'),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=.5
        )
        plt.title('Spearman Correlation')
        
        # Adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Create a table with all statistics
        plt.figure(figsize=(14, len(df) * 0.5 + 1))
        plt.axis('off')
        
        table = plt.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Statistical Test Results', fontsize=16)
        plt.tight_layout()
        
        # Save or show the table
        if save_path:
            table_path = save_path.replace('.png', '_table.png')
            plt.savefig(table_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_class_performance_comparison(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        entropies: np.ndarray,
                                        class_names: Optional[List[str]] = None,
                                        title: str = 'Class Performance Comparison',
                                        save_path: Optional[str] = None) -> None:
        """
        Plot a comparison of performance metrics for each class.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            entropies: Entropy values for each observation
            class_names: Names of the classes
            title: Plot title
            save_path: Path to save the plot
        """
        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Determine class names if not provided
        if class_names is None:
            class_names = {cls: f'Class {cls}' for cls in unique_classes}
        elif isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        
        # Calculate metrics for each class
        metrics_data = []
        
        for cls in unique_classes:
            # True positives
            tp = np.sum((y_true == cls) & (y_pred == cls))
            
            # False positives
            fp = np.sum((y_true != cls) & (y_pred == cls))
            
            # False negatives
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            # Calculate precision, recall, and F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Get entropy for correctly classified samples
            correct_indices = np.where((y_true == cls) & (y_pred == cls))[0]
            mean_entropy = np.mean(entropies[correct_indices]) if len(correct_indices) > 0 else 0
            
            # Store metrics
            metrics_data.append({
                'Class': class_names.get(cls, f'Class {cls}'),
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Mean Entropy': mean_entropy,
                'Sample Count': np.sum(y_true == cls)
            })
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        plt.subplot(2, 2, 1)
        sns.barplot(x='Class', y='Precision', data=df)
        plt.title('Precision by Class')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.barplot(x='Class', y='Recall', data=df)
        plt.title('Recall by Class')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        sns.barplot(x='Class', y='F1 Score', data=df)
        plt.title('F1 Score by Class')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='Class', y='Mean Entropy', data=df)
        plt.title('Mean Entropy by Class')
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Create a table with all metrics
        plt.figure(figsize=(12, len(df) * 0.5 + 1))
        plt.axis('off')
        
        table = plt.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Class Performance Metrics', fontsize=16)
        plt.tight_layout()
        
        # Save or show the table
        if save_path:
            table_path = save_path.replace('.png', '_table.png')
            plt.savefig(table_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_similarity_entropy_joint(self, 
                                    average_similarities: Dict[int, np.ndarray],
                                    entropies: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    title: str = 'Joint Distribution of Similarity and Entropy',
                                    save_path: Optional[str] = None) -> None:
        """
        Plot joint distribution of similarity and entropy for each class.
        
        Args:
            average_similarities: Dictionary mapping class labels to average similarity arrays
            entropies: Entropy values for each observation
            y_pred: Predicted class labels
            class_names: Names of the classes
            title: Plot title
            save_path: Path to save the plot
        """
        # Get unique predicted classes
        unique_classes = np.unique(y_pred)
        
        # Determine class names if not provided
        if class_names is None:
            class_names = {cls: f'Class {cls}' for cls in unique_classes}
        elif isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        
        # Calculate number of subplots
        n_classes = len([cls for cls in unique_classes if cls in average_similarities])
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        # Create figure
        plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        
        # Plot joint distribution for each class
        subplot_idx = 1
        for cls in unique_classes:
            # Get indices of observations predicted as this class
            indices = np.where(y_pred == cls)[0]
            
            if len(indices) <= 1 or cls not in average_similarities:
                # Skip classes with only one observation or no similarity data
                continue
            
            # Get average similarity and entropy for this class
            avg_similarity = average_similarities[cls]
            cls_entropies = entropies[indices]
            
            # Create subplot
            plt.subplot(n_rows, n_cols, subplot_idx)
            
            # Create joint plot
            sns.kdeplot(
                x=avg_similarity,
                y=cls_entropies,
                cmap="viridis",
                fill=True,
                thresh=0.05
            )
            
            # Add scatter plot
            plt.scatter(
                avg_similarity,
                cls_entropies,
                alpha=0.6,
                edgecolor='k',
                s=50
            )
            
            # Calculate Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(avg_similarity, cls_entropies)
            
            # Calculate Spearman's rank correlation
            spearman_corr, spearman_p = stats.spearmanr(avg_similarity, cls_entropies)
            
            # Add correlation information
            plt.title(f"{class_names.get(cls, f'Class {cls}')}\nPearson: r={pearson_corr:.2f}, p={pearson_p:.4f}\nSpearman: ρ={spearman_corr:.2f}, p={spearman_p:.4f}")
            plt.xlabel('Average Cosine Similarity')
            plt.ylabel('Entropy')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            subplot_idx += 1
        
        # Adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_misclassification_analysis(self, 
                                      representations: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      entropies: np.ndarray,
                                      class_names: Optional[List[str]] = None,
                                      title: str = 'Misclassification Analysis',
                                      save_path: Optional[str] = None) -> None:
        """
        Plot analysis of misclassified samples.
        
        Args:
            representations: Intermediate representations from the model
            y_true: True class labels
            y_pred: Predicted class labels
            entropies: Entropy values for each observation
            class_names: Names of the classes
            title: Plot title
            save_path: Path to save the plot
        """
        # Get indices of correctly and incorrectly classified samples
        correct_indices = np.where(y_true == y_pred)[0]
        incorrect_indices = np.where(y_true != y_pred)[0]
        
        # Apply dimensionality reduction to representations
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_representations = pca.fit_transform(representations)
        
        # Create a scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot correctly classified samples
        plt.scatter(reduced_representations[correct_indices, 0], reduced_representations[correct_indices, 1], 
                    c='g', label='Correctly Classified', alpha=0.6)
        
        # Plot incorrectly classified samples
        plt.scatter(reduced_representations[incorrect_indices, 0], reduced_representations[incorrect_indices, 1], 
                    c='r', label='Misclassified', alpha=0.6)
        
        # Add entropy as color intensity for misclassified samples
        plt.scatter(reduced_representations[incorrect_indices, 0], reduced_representations[incorrect_indices, 1], 
                    c=entropies[incorrect_indices], cmap='viridis', edgecolor='k', s=50, alpha=0.6)
        
        # Add labels and title
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Entropy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    def run_full_visualization(self,
                             representations: np.ndarray,
                             softmax_outputs: np.ndarray,
                             y_true: np.ndarray,
                             y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Run a full visualization analysis pipeline.
        
        This method performs the following steps:
        1. Plot correlation heatmap between similarity and entropy
        2. Plot class performance comparison
        3. Plot similarity vs entropy scatter plots
        4. Plot entropy distribution
        5. Plot similarity distribution
        
        Args:
            representations: Intermediate representations from the model
            softmax_outputs: Softmax outputs from the model
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary containing all visualization results
        """
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Calculate entropy from softmax outputs
        entropies = np.apply_along_axis(stats.entropy, 1, softmax_outputs, base=2)
        
        # Calculate pairwise cosine similarity between representations
        similarity_matrices = {}
        unique_classes = np.unique(y_pred)
        
        for cls in unique_classes:
            # Get indices of observations predicted as this class
            indices = np.where(y_pred == cls)[0]
            
            if len(indices) <= 1:
                # Skip classes with only one observation
                continue
            
            # Get representations for this class
            class_representations = representations[indices]
            
            # Calculate pairwise cosine similarity
            n_samples = class_representations.shape[0]
            similarity_matrix = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                for j in range(i, n_samples):
                    # Calculate cosine similarity (1 - cosine distance)
                    sim = 1 - cosine(class_representations[i], class_representations[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
            
            # Store similarity matrix
            similarity_matrices[cls] = similarity_matrix
        
        # 1. Plot correlation heatmap
        print("Plotting correlation heatmap...")
        self.plot_correlation_heatmap(
            similarity_matrices=similarity_matrices,
            entropies=entropies,
            y_pred=y_pred,
            title='Correlation between Similarity and Entropy',
            save_path=os.path.join(self.save_dir, 'correlation_heatmap.png')
        )
        
        # 2. Plot class performance comparison
        print("Plotting class performance comparison...")
        self.plot_class_performance_comparison(
            y_true=y_true,
            y_pred=y_pred,
            entropies=entropies,
            title='Class Performance Comparison',
            save_path=os.path.join(self.save_dir, 'class_performance.png')
        )
        
        # 3. Plot similarity vs entropy scatter plots
        print("Plotting similarity vs entropy scatter plots...")
        self.plot_similarity_entropy_scatter(
            similarity_matrices=similarity_matrices,
            entropies=entropies,
            y_pred=y_pred,
            title='Similarity vs Entropy Scatter Plots',
            save_path=os.path.join(self.save_dir, 'similarity_entropy_scatter.png')
        )
        
        # 4. Plot entropy distribution
        print("Plotting entropy distribution...")
        self.plot_entropy_distribution(
            entropies=entropies,
            y_true=y_true,
            y_pred=y_pred,
            title='Entropy Distribution',
            save_path=os.path.join(self.save_dir, 'entropy_distribution.png')
        )
        
        # 5. Plot similarity distribution
        print("Plotting similarity distribution...")
        self.plot_similarity_distribution(
            similarity_matrices=similarity_matrices,
            y_pred=y_pred,
            title='Similarity Distribution',
            save_path=os.path.join(self.save_dir, 'similarity_distribution.png')
        )
        
        # Compile results
        results = {
            'entropies': entropies,
            'similarity_matrices': similarity_matrices,
            'correlation_stats': self._calculate_correlation_stats(similarity_matrices, entropies, y_pred)
        }
        
        return results
    
    def _calculate_correlation_stats(self,
                                   similarity_matrices: Dict[int, np.ndarray],
                                   entropies: np.ndarray,
                                   y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Calculate correlation statistics between similarity and entropy for each class.
        
        Args:
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            entropies: Entropy values for each observation
            y_pred: Predicted class labels
            
        Returns:
            Dictionary mapping class labels to correlation statistics
        """
        # Get unique predicted classes
        unique_classes = np.unique(y_pred)
        
        # Dictionary to store correlation statistics for each class
        correlation_stats = {}
        
        # Calculate average similarity for each class
        average_similarities = {}
        for cls, similarity_matrix in similarity_matrices.items():
            n_samples = similarity_matrix.shape[0]
            avg_similarity = np.zeros(n_samples)
            
            for i in range(n_samples):
                similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i+1:]
                ])
                avg_similarity[i] = np.mean(similarities)
            
            average_similarities[cls] = avg_similarity
        
        # Calculate correlation for each class
        for cls in unique_classes:
            # Get indices of observations predicted as this class
            indices = np.where(y_pred == cls)[0]
            
            if len(indices) <= 1 or cls not in average_similarities:
                # Skip classes with only one observation or no similarity data
                continue
            
            # Get average similarity and entropy for this class
            avg_similarity = average_similarities[cls]
            cls_entropies = entropies[indices]
            
            # Calculate Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(avg_similarity, cls_entropies)
            
            # Calculate Spearman's rank correlation
            spearman_corr, spearman_p = stats.spearmanr(avg_similarity, cls_entropies)
            
            # Store correlation statistics
            correlation_stats[cls] = {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'sample_size': len(indices)
            }
        
        return correlation_stats
    
    def plot_similarity_entropy_scatter(self,
                                    similarity_matrices: Dict[int, np.ndarray],
                                    entropies: np.ndarray,
                                    y_pred: np.ndarray,
                                    title: str = 'Similarity vs Entropy Scatter Plots',
                                    save_path: Optional[str] = None) -> None:
        """
        Plot scatter plots of average similarity vs entropy for each class.
        
        Args:
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            entropies: Entropy values for each observation
            y_pred: Predicted class labels
            title: Plot title
            save_path: Path to save the plot
        """
        # Get unique predicted classes
        unique_classes = np.unique(y_pred)
        
        # Calculate average similarity for each class
        average_similarities = {}
        for cls, similarity_matrix in similarity_matrices.items():
            n_samples = similarity_matrix.shape[0]
            avg_similarity = np.zeros(n_samples)
            
            for i in range(n_samples):
                similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i+1:]
                ])
                avg_similarity[i] = np.mean(similarities)
            
            average_similarities[cls] = avg_similarity
        
        # Determine number of subplots
        n_classes = len(average_similarities)
        if n_classes == 0:
            return
        
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        # Create figure
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        # Get the maximum entropy value
        max_entropy = np.max(entropies)
        
        # Plot scatter plots for each class
        for i, cls in enumerate(sorted(average_similarities.keys())):
            # Get indices of observations predicted as this class
            indices = np.where(y_pred == cls)[0]
            
            # Get average similarity and entropy for this class
            avg_similarity = average_similarities[cls]
            cls_entropies = entropies[indices]
            
            # Calculate correlation
            pearson_corr, pearson_p = stats.pearsonr(avg_similarity, cls_entropies)
            spearman_corr, spearman_p = stats.spearmanr(avg_similarity, cls_entropies)
            
            # Create subplot
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Plot scatter plot
            plt.scatter(avg_similarity, cls_entropies, alpha=0.7)
            
            # Add regression line
            x = np.linspace(0, 1, 100)
            slope, intercept = np.polyfit(avg_similarity, cls_entropies, 1)
            plt.plot(x, slope * x + intercept, 'r--')
            
            # Add correlation information
            plt.title(f'Class {cls}')
            plt.xlabel('Average Cosine Similarity')
            plt.ylabel('Entropy')
            plt.text(0.05, 0.95, f'Pearson: {pearson_corr:.2f} (p={pearson_p:.3f})\nSpearman: {spearman_corr:.2f} (p={spearman_p:.3f})',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set axis limits
            plt.xlim(0, 1)
            plt.ylim(0, max_entropy)
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_entropy_distribution(self,
                                entropies: np.ndarray,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                title: str = 'Entropy Distribution',
                                save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of entropy values for correct and incorrect predictions.
        
        Args:
            entropies: Entropy values for each observation
            y_true: True class labels
            y_pred: Predicted class labels
            title: Plot title
            save_path: Path to save the plot
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Get indices of correct and incorrect predictions
        correct_indices = np.where(y_true == y_pred)[0]
        incorrect_indices = np.where(y_true != y_pred)[0]
        
        # Plot histograms
        plt.subplot(1, 2, 1)
        plt.hist(entropies, bins=20, alpha=0.7, label='All')
        if len(correct_indices) > 0:
            plt.hist(entropies[correct_indices], bins=20, alpha=0.7, label='Correct')
        if len(incorrect_indices) > 0:
            plt.hist(entropies[incorrect_indices], bins=20, alpha=0.7, label='Incorrect')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        plt.title('Entropy Histogram')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot box plots
        plt.subplot(1, 2, 2)
        data = []
        labels = []
        
        data.append(entropies)
        labels.append('All')
        
        if len(correct_indices) > 0:
            data.append(entropies[correct_indices])
            labels.append('Correct')
        
        if len(incorrect_indices) > 0:
            data.append(entropies[incorrect_indices])
            labels.append('Incorrect')
        
        plt.boxplot(data, labels=labels)
        plt.ylabel('Entropy')
        plt.title('Entropy Box Plot')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_similarity_distribution(self,
                                   similarity_matrices: Dict[int, np.ndarray],
                                   y_pred: np.ndarray,
                                   title: str = 'Similarity Distribution',
                                   save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of average similarity values for each class.
        
        Args:
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            y_pred: Predicted class labels
            title: Plot title
            save_path: Path to save the plot
        """
        # Calculate average similarity for each class
        average_similarities = {}
        for cls, similarity_matrix in similarity_matrices.items():
            n_samples = similarity_matrix.shape[0]
            avg_similarity = np.zeros(n_samples)
            
            for i in range(n_samples):
                similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i+1:]
                ])
                avg_similarity[i] = np.mean(similarities)
            
            average_similarities[cls] = avg_similarity
        
        # Check if there are any classes with similarity data
        if not average_similarities:
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot histograms for each class
        plt.subplot(1, 2, 1)
        for cls, avg_similarity in average_similarities.items():
            plt.hist(avg_similarity, bins=10, alpha=0.7, label=f'Class {cls}')
        plt.xlabel('Average Cosine Similarity')
        plt.ylabel('Count')
        plt.title('Similarity Histogram by Class')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot box plots for each class
        plt.subplot(1, 2, 2)
        data = []
        labels = []
        
        for cls, avg_similarity in average_similarities.items():
            data.append(avg_similarity)
            labels.append(f'Class {cls}')
        
        plt.boxplot(data, labels=labels)
        plt.ylabel('Average Cosine Similarity')
        plt.title('Similarity Box Plot by Class')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
