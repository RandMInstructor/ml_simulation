"""
Residuals Analysis Module for ML Simulation Environment.

This module provides functionality for analyzing model residuals,
including confusion matrix generation, intermediate representation analysis,
cosine similarity calculation, and entropy analysis for image classification.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from scipy.spatial.distance import cosine, pdist, squareform
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import os


class ResidualsAnalyzer:
    """
    A class for analyzing model residuals in multi-class classification problems.
    
    This class provides functionality for generating confusion matrices,
    extracting intermediate representations, calculating pairwise cosine similarity,
    and analyzing entropy from softmax outputs, with a focus on image classification.
    """
    
    def __init__(self, save_dir: str = './results'):
        """
        Initialize the ResidualsAnalyzer.
        
        Args:
            save_dir: Directory to save analysis results
        """
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Dictionary to store analysis results
        self.results = {}
    
    def generate_confusion_matrix(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                normalize: Optional[str] = None) -> np.ndarray:
        """
        Generate a confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization option ('true', 'pred', 'all', or None)
            
        Returns:
            Confusion matrix
        """
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        # Store in results
        self.results['confusion_matrix'] = cm
        
        return cm
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             normalize: Optional[str] = None,
                             title: str = 'Confusion Matrix',
                             cmap: str = 'Blues',
                             save_path: Optional[str] = None) -> None:
        """
        Plot the confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            normalize: Normalization option ('true', 'pred', 'all', or None)
            title: Plot title
            cmap: Colormap
            save_path: Path to save the plot
        """
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_true, y_pred, normalize=normalize)
        
        # Determine class names if not provided
        if class_names is None:
            n_classes = cm.shape[0]
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(title, fontsize=16)
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def calculate_cosine_similarity(self, 
                                  representations: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Calculate pairwise cosine similarity between representations of observations
        predicted to be in the same class.
        
        Args:
            representations: Intermediate representations from the model
            y_pred: Predicted class labels
            
        Returns:
            Dictionary mapping class labels to similarity matrices
        """
        # Get unique predicted classes
        unique_classes = np.unique(y_pred)
        
        # Dictionary to store similarity matrices for each class
        similarity_matrices = {}
        
        # Calculate similarity matrix for each class
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
        
        # Store in results
        self.results['cosine_similarity'] = similarity_matrices
        
        return similarity_matrices
    
    def calculate_average_similarity(self, 
                                   similarity_matrices: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Calculate average cosine similarity for each observation with all other
        observations in the same predicted class.
        
        Args:
            similarity_matrices: Dictionary mapping class labels to similarity matrices
            
        Returns:
            Dictionary mapping class labels to average similarity arrays
        """
        # Dictionary to store average similarity for each class
        average_similarities = {}
        
        # Calculate average similarity for each class
        for cls, similarity_matrix in similarity_matrices.items():
            # Calculate average similarity for each observation
            # Exclude self-similarity (diagonal elements)
            n_samples = similarity_matrix.shape[0]
            avg_similarity = np.zeros(n_samples)
            
            for i in range(n_samples):
                # Get all similarities except self-similarity
                similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i+1:]
                ])
                
                # Calculate average
                avg_similarity[i] = np.mean(similarities)
            
            # Store average similarity
            average_similarities[cls] = avg_similarity
        
        # Store in results
        self.results['average_similarity'] = average_similarities
        
        return average_similarities
    
    def calculate_entropy(self, 
                        softmax_outputs: np.ndarray) -> np.ndarray:
        """
        Calculate entropy from softmax outputs.
        
        Args:
            softmax_outputs: Softmax outputs from the model
            
        Returns:
            Entropy values for each observation
        """
        # Calculate entropy for each observation
        entropies = np.apply_along_axis(entropy, 1, softmax_outputs)
        
        # Store in results
        self.results['entropy'] = entropies
        
        return entropies
    
    def analyze_correlation(self, 
                          average_similarities: Dict[int, np.ndarray],
                          entropies: np.ndarray,
                          y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Analyze correlation between average similarity and entropy for each class.
        
        Args:
            average_similarities: Dictionary mapping class labels to average similarity arrays
            entropies: Entropy values for each observation
            y_pred: Predicted class labels
            
        Returns:
            Dictionary mapping class labels to correlation statistics
        """
        # Get unique predicted classes
        unique_classes = np.unique(y_pred)
        
        # Dictionary to store correlation statistics for each class
        correlation_stats = {}
        
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
            
            # Calculate correlation
            correlation = np.corrcoef(avg_similarity, cls_entropies)[0, 1]
            
            # Calculate p-value using t-test
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(avg_similarity, cls_entropies)
            
            # Store correlation statistics
            correlation_stats[cls] = {
                'correlation': correlation,
                'p_value': p_value
            }
        
        # Store in results
        self.results['correlation_stats'] = correlation_stats
        
        return correlation_stats
    
    def plot_correlation(self, 
                       average_similarities: Dict[int, np.ndarray],
                       entropies: np.ndarray,
                       y_pred: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       title: str = 'Correlation between Similarity and Entropy',
                       save_path: Optional[str] = None) -> None:
        """
        Plot correlation between average similarity and entropy for each class.
        
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
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        # Plot correlation for each class
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
            
            # Plot scatter plot
            plt.scatter(avg_similarity, cls_entropies, alpha=0.7)
            
            # Add trend line
            z = np.polyfit(avg_similarity, cls_entropies, 1)
            p = np.poly1d(z)
            plt.plot(avg_similarity, p(avg_similarity), "r--", alpha=0.7)
            
            # Calculate correlation
            correlation = np.corrcoef(avg_similarity, cls_entropies)[0, 1]
            
            # Calculate p-value using t-test
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(avg_similarity, cls_entropies)
            
            # Add correlation information
            plt.title(f"{class_names.get(cls, f'Class {cls}')} (r={correlation:.2f}, p={p_value:.4f})")
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
    
    def visualize_representations(self, 
                                representations: np.ndarray,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                method: str = 'tsne',
                                perplexity: int = 30,
                                n_components: int = 2,
                                class_names: Optional[List[str]] = None,
                                title: str = 'Intermediate Representations',
                                save_path: Optional[str] = None) -> None:
        """
        Visualize intermediate representations using dimensionality reduction.
        
        Args:
            representations: Intermediate representations from the model
            y_true: True class labels
            y_pred: Predicted class labels
            method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
            perplexity: Perplexity parameter for t-SNE
            n_components: Number of components for dimensionality reduction
            class_names: Names of the classes
            title: Plot title
            save_path: Path to save the plot
        """
        # Determine class names if not provided
        if class_names is None:
            n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        elif method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            except ImportError:
                print("UMAP not installed. Using t-SNE instead.")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Reduce dimensions
        reduced_representations = reducer.fit_transform(representations)
        
        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            indices = np.where(y_true == i)
            plt.scatter(reduced_representations[indices, 0], reduced_representations[indices, 1], label=class_name, alpha=0.6)
        
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    def run_full_analysis(self,
                        representations: np.ndarray,
                        softmax_outputs: np.ndarray,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a full residuals analysis pipeline.
        
        This method performs the following steps:
        1. Generate and plot confusion matrix
        2. Calculate pairwise cosine similarity between representations
        3. Calculate average similarity for each observation
        4. Calculate entropy from softmax outputs
        5. Analyze correlation between similarity and entropy
        
        Args:
            representations: Intermediate representations from the model
            softmax_outputs: Softmax outputs from the model
            y_true: True class labels
            y_pred: Predicted class labels
            save_dir: Directory to save analysis results (if None, uses self.save_dir)
            
        Returns:
            Dictionary containing all analysis results
        """
        # Set save directory
        if save_dir is None:
            save_dir = self.save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Generate and plot confusion matrix
        print("Generating confusion matrix...")
        self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            normalize='true',
            title='Confusion Matrix (Normalized)',
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # 2. Calculate pairwise cosine similarity
        print("Calculating pairwise cosine similarity...")
        similarity_matrices = self.calculate_cosine_similarity(
            representations=representations,
            y_pred=y_pred
        )
        
        # 3. Calculate average similarity
        print("Calculating average similarity...")
        average_similarities = self.calculate_average_similarity(
            similarity_matrices=similarity_matrices
        )
        
        # 4. Calculate entropy
        print("Calculating entropy from softmax outputs...")
        entropies = self.calculate_entropy(
            softmax_outputs=softmax_outputs
        )
        
        # 5. Analyze correlation
        print("Analyzing correlation between similarity and entropy...")
        correlation_stats = self.analyze_correlation(
            average_similarities=average_similarities,
            entropies=entropies,
            y_pred=y_pred
        )
        
        # Compile all results
        results = {
            'confusion_matrix': self.results.get('confusion_matrix'),
            'cosine_similarity': similarity_matrices,
            'average_similarity': average_similarities,
            'entropy': entropies,
            'correlation_stats': correlation_stats
        }
        
        # Save results summary
        self._save_results_summary(
            results=results,
            save_path=os.path.join(save_dir, 'results_summary.txt')
        )
        
        return results
    
    def _save_results_summary(self,
                             results: Dict[str, Any],
                             save_path: str) -> None:
        """
        Save a summary of the analysis results to a text file.
        
        Args:
            results: Dictionary containing analysis results
            save_path: Path to save the summary
        """
        with open(save_path, 'w') as f:
            f.write("=== Residuals Analysis Results Summary ===\n\n")
            
            # Correlation statistics
            f.write("Correlation between Average Similarity and Entropy:\n")
            f.write("-" * 50 + "\n")
            
            correlation_stats = results.get('correlation_stats', {})
            for cls, stats in correlation_stats.items():
                f.write(f"Class {cls}:\n")
                f.write(f"  Pearson correlation: {stats.get('correlation', 'N/A'):.4f}\n")
                f.write(f"  p-value: {stats.get('p_value', 'N/A'):.4f}\n")
                f.write(f"  Significance: {'Significant' if stats.get('p_value', 1.0) < 0.05 else 'Not significant'}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 50 + "\n")
            
            # Calculate average entropy
            entropies = results.get('entropy', np.array([]))
            if len(entropies) > 0:
                f.write(f"Average entropy: {np.mean(entropies):.4f}\n")
                f.write(f"Min entropy: {np.min(entropies):.4f}\n")
                f.write(f"Max entropy: {np.max(entropies):.4f}\n\n")
            
            # Calculate average similarity across all classes
            avg_similarities = []
            for cls, similarities in results.get('average_similarity', {}).items():
                avg_similarities.extend(similarities)
            
            if len(avg_similarities) > 0:
                avg_similarities = np.array(avg_similarities)
                f.write(f"Average cosine similarity: {np.mean(avg_similarities):.4f}\n")
                f.write(f"Min cosine similarity: {np.min(avg_similarities):.4f}\n")
                f.write(f"Max cosine similarity: {np.max(avg_similarities):.4f}\n")
