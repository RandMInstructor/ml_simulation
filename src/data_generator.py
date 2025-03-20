"""
Data Generation Module for ML Simulation Environment.

This module provides customizable methods to generate synthetic data
for multi-class classification problems with various distributions.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional, List, Union, Callable


class DataGenerator:
    """
    A class for generating synthetic data for multi-class classification problems.
    
    This class provides various methods to generate synthetic data with
    customizable parameters for different distributions and class structures.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the DataGenerator.
        
        Args:
            random_state: Seed for random number generation for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_data(self, 
                      method: str = 'blobs', 
                      n_samples: int = 1000, 
                      n_classes: int = 3, 
                      n_features: int = 2,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data using the specified method.
        
        Args:
            method: The method to use for data generation
                   ('blobs', 'classification', 'moons', 'circles', 'gaussian_mixture')
            n_samples: Number of samples to generate
            n_classes: Number of classes
            n_features: Number of features
            **kwargs: Additional parameters specific to the chosen method
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        if method == 'blobs':
            return self._generate_blobs(n_samples, n_classes, n_features, **kwargs)
        elif method == 'classification':
            return self._generate_classification(n_samples, n_classes, n_features, **kwargs)
        elif method == 'moons':
            return self._generate_moons(n_samples, n_classes, **kwargs)
        elif method == 'circles':
            return self._generate_circles(n_samples, n_classes, **kwargs)
        elif method == 'gaussian_mixture':
            return self._generate_gaussian_mixture(n_samples, n_classes, n_features, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_blobs(self, 
                       n_samples: int, 
                       n_classes: int, 
                       n_features: int,
                       centers: Optional[Union[int, np.ndarray]] = None,
                       cluster_std: Union[float, List[float]] = 1.0,
                       center_box: Tuple[float, float] = (-10.0, 10.0),
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate isotropic Gaussian blobs for clustering.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (centers)
            n_features: Number of features
            centers: Centers of the blobs
            cluster_std: Standard deviation of the blobs
            center_box: Bounding box for each cluster center
            **kwargs: Additional parameters for make_blobs
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        if centers is None:
            centers = n_classes
            
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=self.random_state,
            **kwargs
        )
        
        return X, y
    
    def _generate_classification(self, 
                                n_samples: int, 
                                n_classes: int, 
                                n_features: int,
                                n_informative: int = None,
                                n_redundant: int = 0,
                                n_repeated: int = 0,
                                class_sep: float = 1.0,
                                **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random n-class classification problem.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            n_features: Number of features
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_repeated: Number of repeated features
            class_sep: Factor multiplying the hypercube size
            **kwargs: Additional parameters for make_classification
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        if n_informative is None:
            n_informative = max(2, n_features // 2)
            
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            class_sep=class_sep,
            random_state=self.random_state,
            **kwargs
        )
        
        return X, y
    
    def _generate_moons(self, 
                       n_samples: int, 
                       n_classes: int,
                       noise: float = 0.1,
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two interleaving half circles.
        
        Note: This method only supports binary classification (n_classes=2).
        For more classes, it creates multiple pairs of moons and assigns labels accordingly.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (must be even)
            noise: Standard deviation of Gaussian noise added to the data
            **kwargs: Additional parameters for make_moons
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        if n_classes != 2 and n_classes % 2 != 0:
            raise ValueError("For 'moons' method, n_classes must be 2 or an even number")
            
        if n_classes == 2:
            X, y = make_moons(
                n_samples=n_samples,
                noise=noise,
                random_state=self.random_state,
                **kwargs
            )
        else:
            # Create multiple pairs of moons for more than 2 classes
            samples_per_pair = n_samples // (n_classes // 2)
            X_list, y_list = [], []
            
            for i in range(n_classes // 2):
                X_pair, y_pair = make_moons(
                    n_samples=samples_per_pair,
                    noise=noise,
                    random_state=self.random_state + i,
                    **kwargs
                )
                
                # Shift each pair to avoid overlap
                shift = np.array([4 * i, 4 * i])
                X_pair = X_pair + shift
                
                # Adjust labels for this pair
                y_pair = y_pair + 2 * i
                
                X_list.append(X_pair)
                y_list.append(y_pair)
            
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            
            # Shuffle the data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        return X, y
    
    def _generate_circles(self, 
                         n_samples: int, 
                         n_classes: int,
                         noise: float = 0.1,
                         factor: float = 0.8,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate concentric circles.
        
        Note: This method only supports binary classification (n_classes=2).
        For more classes, it creates multiple pairs of circles and assigns labels accordingly.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (must be even)
            noise: Standard deviation of Gaussian noise added to the data
            factor: Scale factor between inner and outer circle
            **kwargs: Additional parameters for make_circles
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        if n_classes != 2 and n_classes % 2 != 0:
            raise ValueError("For 'circles' method, n_classes must be 2 or an even number")
            
        if n_classes == 2:
            X, y = make_circles(
                n_samples=n_samples,
                noise=noise,
                factor=factor,
                random_state=self.random_state,
                **kwargs
            )
        else:
            # Create multiple pairs of circles for more than 2 classes
            samples_per_pair = n_samples // (n_classes // 2)
            X_list, y_list = [], []
            
            for i in range(n_classes // 2):
                X_pair, y_pair = make_circles(
                    n_samples=samples_per_pair,
                    noise=noise,
                    factor=factor,
                    random_state=self.random_state + i,
                    **kwargs
                )
                
                # Shift each pair to avoid overlap
                shift = np.array([4 * i, 4 * i])
                X_pair = X_pair + shift
                
                # Adjust labels for this pair
                y_pair = y_pair + 2 * i
                
                X_list.append(X_pair)
                y_list.append(y_pair)
            
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            
            # Shuffle the data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        
        return X, y
    
    def _generate_gaussian_mixture(self, 
                                  n_samples: int, 
                                  n_classes: int, 
                                  n_features: int,
                                  means: Optional[np.ndarray] = None,
                                  covs: Optional[List[np.ndarray]] = None,
                                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from a Gaussian mixture model.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (mixture components)
            n_features: Number of features
            means: Means of the Gaussian components (default: random)
            covs: Covariance matrices of the Gaussian components (default: random)
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        # Generate random means if not provided
        if means is None:
            means = np.random.uniform(-10, 10, size=(n_classes, n_features))
        
        # Generate random covariance matrices if not provided
        if covs is None:
            covs = []
            for _ in range(n_classes):
                # Create a random positive definite matrix
                A = np.random.randn(n_features, n_features)
                cov = np.dot(A, A.T) + np.eye(n_features)  # Ensure positive definiteness
                covs.append(cov)
        
        # Determine samples per class
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        X_list, y_list = [], []
        
        for i in range(n_classes):
            # Adjust for remainder
            n_samples_i = samples_per_class + (1 if i < remainder else 0)
            
            # Generate samples from multivariate normal distribution
            X_i = np.random.multivariate_normal(
                mean=means[i],
                cov=covs[i],
                size=n_samples_i
            )
            
            y_i = np.full(n_samples_i, i)
            
            X_list.append(X_i)
            y_list.append(y_i)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def preprocess_data(self, X: np.ndarray, 
                       scale: bool = True,
                       **kwargs) -> np.ndarray:
        """
        Preprocess the generated data.
        
        Args:
            X: Input features
            scale: Whether to standardize the features
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed features
        """
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return X
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, 
                  val_size: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            
        Returns:
            Dictionary containing the split data
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_samples = int(n_samples * test_size)
        val_samples = int(n_samples * val_size)
        train_samples = n_samples - test_samples - val_samples
        
        train_indices = indices[:train_samples]
        val_indices = indices[train_samples:train_samples + val_samples] if val_size > 0 else None
        test_indices = indices[-test_samples:]
        
        result = {
            'X_train': X[train_indices],
            'y_train': y[train_indices],
            'X_test': X[test_indices],
            'y_test': y[test_indices]
        }
        
        if val_size > 0:
            result['X_val'] = X[val_indices]
            result['y_val'] = y[val_indices]
        
        return result
    
    def visualize_data(self, X: np.ndarray, y: np.ndarray, 
                      title: str = 'Data Visualization',
                      save_path: Optional[str] = None) -> None:
        """
        Visualize the generated data.
        
        Args:
            X: Input features
            y: Target labels
            title: Plot title
            save_path: Path to save the visualization (if None, display only)
        """
        if X.shape[1] > 2:
            # For high-dimensional data, use PCA to reduce to 2D for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
        
        plt.figure(figsize=(10, 8))
        
        # Get unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Create a colormap
        cmap = plt.cm.get_cmap('tab10', n_classes)
        
        # Plot each class
        for i, cls in enumerate(unique_classes):
            plt.scatter(
                X_2d[y == cls, 0],
                X_2d[y == cls, 1],
                c=[cmap(i)],
                label=f'Class {cls}',
                alpha=0.7,
                edgecolors='k'
            )
        
        p<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>