"""
Image Data Generation Module for ML Simulation Environment.

This module provides customizable methods to generate synthetic image data
for multi-class classification problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional, List, Union, Callable
import os
from sklearn.model_selection import train_test_split


class ImageDataGenerator:
    """
    A class for generating synthetic image data for multi-class classification problems.
    
    This class provides various methods to generate synthetic images with
    customizable parameters for different patterns and class structures.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ImageDataGenerator.
        
        Args:
            random_state: Seed for random number generation for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_image_data(self, 
                          method: str = 'shapes', 
                          n_samples: int = 1000, 
                          n_classes: int = 3,
                          image_size: Tuple[int, int] = (32, 32),
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic image data using the specified method.
        
        Args:
            method: The method to use for data generation
                   ('shapes', 'patterns', 'noise', 'gradients')
            n_samples: Number of samples to generate
            n_classes: Number of classes
            image_size: Size of the generated images (height, width)
            **kwargs: Additional parameters specific to the chosen method
            
        Returns:
            Tuple containing images (X) and labels (y)
        """
        if method == 'shapes':
            return self._generate_shapes(n_samples, n_classes, image_size, **kwargs)
        elif method == 'patterns':
            return self._generate_patterns(n_samples, n_classes, image_size, **kwargs)
        elif method == 'noise':
            return self._generate_noise(n_samples, n_classes, image_size, **kwargs)
        elif method == 'gradients':
            return self._generate_gradients(n_samples, n_classes, image_size, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_shapes(self, 
                        n_samples: int, 
                        n_classes: int,
                        image_size: Tuple[int, int] = (32, 32),
                        shape_types: Optional[List[str]] = None,
                        background_noise: float = 0.05,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate images with different shapes for each class.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            image_size: Size of the generated images (height, width)
            shape_types: List of shape types to use (default: ['circle', 'square', 'triangle', 'cross', 'ellipse'])
            background_noise: Standard deviation of background noise
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing images (X) and labels (y)
        """
        height, width = image_size
        
        # Default shape types
        if shape_types is None:
            shape_types = ['circle', 'square', 'triangle', 'cross', 'ellipse']
        
        # Ensure we have enough shape types for the number of classes
        if len(shape_types) < n_classes:
            # Repeat shape types if necessary
            shape_types = (shape_types * ((n_classes // len(shape_types)) + 1))[:n_classes]
        
        # Initialize images and labels with channel dimension
        X = np.zeros((n_samples, height, width, 1))
        y = np.zeros(n_samples, dtype=int)
        
        # Samples per class
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        sample_idx = 0
        for class_idx in range(n_classes):
            # Adjust for remainder
            n_samples_i = samples_per_class + (1 if class_idx < remainder else 0)
            
            # Get shape type for this class
            shape_type = shape_types[class_idx]
            
            for i in range(n_samples_i):
                # Create image with background noise
                img = np.random.normal(0, background_noise, (height, width))
                
                # Random position and size for the shape
                center_x = np.random.randint(width // 4, 3 * width // 4)
                center_y = np.random.randint(height // 4, 3 * height // 4)
                size = np.random.randint(min(width, height) // 6, min(width, height) // 3)
                
                # Draw shape
                if shape_type == 'circle':
                    # Create a circle
                    y_indices, x_indices = np.ogrid[:height, :width]
                    dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                    mask = dist_from_center <= size
                    img[mask] = 1.0
                
                elif shape_type == 'square':
                    # Create a square
                    x_min = max(0, center_x - size)
                    x_max = min(width, center_x + size)
                    y_min = max(0, center_y - size)
                    y_max = min(height, center_y + size)
                    img[y_min:y_max, x_min:x_max] = 1.0
                
                elif shape_type == 'triangle':
                    # Create a triangle
                    for y_coord in range(height):
                        for x_coord in range(width):
                            # Check if point is inside triangle
                            x1, y1 = center_x, center_y - size
                            x2, y2 = center_x - size, center_y + size
                            x3, y3 = center_x + size, center_y + size
                            
                            # Barycentric coordinates
                            denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                            a = ((y2 - y3) * (x_coord - x3) + (x3 - x2) * (y_coord - y3)) / denominator
                            b = ((y3 - y1) * (x_coord - x3) + (x1 - x3) * (y_coord - y3)) / denominator
                            c = 1 - a - b
                            
                            if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
                                img[y_coord, x_coord] = 1.0
                
                elif shape_type == 'cross':
                    # Create a cross
                    thickness = size // 3
                    
                    # Horizontal line
                    y_min = max(0, center_y - thickness // 2)
                    y_max = min(height, center_y + thickness // 2)
                    x_min = max(0, center_x - size)
                    x_max = min(width, center_x + size)
                    img[y_min:y_max, x_min:x_max] = 1.0
                    
                    # Vertical line
                    y_min = max(0, center_y - size)
                    y_max = min(height, center_y + size)
                    x_min = max(0, center_x - thickness // 2)
                    x_max = min(width, center_x + thickness // 2)
                    img[y_min:y_max, x_min:x_max] = 1.0
                
                elif shape_type == 'ellipse':
                    # Create an ellipse
                    y_indices, x_indices = np.ogrid[:height, :width]
                    # Random aspect ratio
                    aspect_ratio = np.random.uniform(0.5, 2.0)
                    a = size  # semi-major axis
                    b = size * aspect_ratio  # semi-minor axis
                    
                    # Random rotation
                    angle = np.random.uniform(0, 2 * np.pi)
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    
                    # Rotated coordinates
                    x_rot = cos_angle * (x_indices - center_x) + sin_angle * (y_indices - center_y)
                    y_rot = -sin_angle * (x_indices - center_x) + cos_angle * (y_indices - center_y)
                    
                    # Ellipse equation
                    mask = ((x_rot / a)**2 + (y_rot / b)**2) <= 1
                    img[mask] = 1.0
                
                # Add random variations
                img += np.random.normal(0, 0.1, (height, width))
                
                # Clip values to [0, 1]
                img = np.clip(img, 0, 1)
                
                # Store image and label with channel dimension
                X[sample_idx, :, :, 0] = img
                y[sample_idx] = class_idx
                sample_idx += 1
        
        return X, y
    
    def _generate_patterns(self, 
                          n_samples: int, 
                          n_classes: int,
                          image_size: Tuple[int, int] = (32, 32),
                          pattern_types: Optional[List[str]] = None,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate images with different patterns for each class.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            image_size: Size of the generated images (height, width)
            pattern_types: List of pattern types to use (default: ['grid', 'stripes', 'dots', 'waves'])
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing images (X) and labels (y)
        """
        height, width = image_size
        
        # Default pattern types
        if pattern_types is None:
            pattern_types = ['grid', 'stripes', 'dots', 'waves']
        
        # Ensure we have enough pattern types for the number of classes
        if len(pattern_types) < n_classes:
            # Repeat pattern types if necessary
            pattern_types = (pattern_types * ((n_classes // len(pattern_types)) + 1))[:n_classes]
        
        # Initialize images and labels with channel dimension
        X = np.zeros((n_samples, height, width, 1))
        y = np.zeros(n_samples, dtype=int)
        
        # Samples per class
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        sample_idx = 0
        for class_idx in range(n_classes):
            # Adjust for remainder
            n_samples_i = samples_per_class + (1 if class_idx < remainder else 0)
            
            # Get pattern type for this class
            pattern_type = pattern_types[class_idx]
            
            for i in range(n_samples_i):
                # Create base image
                img = np.zeros((height, width))
                
                # Random parameters
                frequency = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                
                if pattern_type == 'grid':
                    # Create a grid pattern
                    grid_size = np.random.randint(4, 10)
                    for i in range(0, height, grid_size):
                        img[i:i+1, :] = 1.0
                    for j in range(0, width, grid_size):
                        img[:, j:j+1] = 1.0
                
                elif pattern_type == 'stripes':
                    # Create stripes pattern
                    stripe_width = np.random.randint(2, 6)
                    for i in range(0, height, 2 * stripe_width):
                        img[i:i+stripe_width, :] = 1.0
                
                elif pattern_type == 'dots':
                    # Create dots pattern
                    n_dots = np.random.randint(10, 30)
                    dot_size = np.random.randint(1, 4)
                    
                    for _ in range(n_dots):
                        x = np.random.randint(0, width)
                        y = np.random.randint(0, height)
                        
                        # Create a circular dot
                        y_indices, x_indices = np.ogrid[:height, :width]
                        dist_from_center = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
                        mask = dist_from_center <= dot_size
                        img[mask] += 1.0
                
                elif pattern_type == 'waves':
                    # Sinusoidal waves
                    y_indices, x_indices = np.ogrid[:height, :width]
                    
                    # Create wave pattern
                    waves = np.sin(2 * np.pi * frequency * x_indices / width + phase) * \
                            np.sin(2 * np.pi * frequency * y_indices / height + phase)
                    
                    # Normalize to [0, 1]
                    waves = (waves + 1) / 2
                    
                    img += waves
                
                # Add random variations
                img += np.random.normal(0, 0.1, (height, width))
                
                # Clip values to [0, 1]
                img = np.clip(img, 0, 1)
                
                # Store image and label with channel dimension
                X[sample_idx, :, :, 0] = img
                y[sample_idx] = class_idx
                sample_idx += 1
        
        return X, y
    
    def _generate_noise(self, 
                       n_samples: int, 
                       n_classes: int,
                       image_size: Tuple[int, int] = (32, 32),
                       noise_types: Optional[List[str]] = None,
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate images with different types of noise for each class.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            image_size: Size of the generated images (height, width)
            noise_types: List of noise types to use (default: ['gaussian', 'salt_pepper', 'perlin'])
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing images (X) and labels (y)
        """
        height, width = image_size
        
        # Default noise types
        if noise_types is None:
            noise_types = ['gaussian', 'salt_pepper', 'uniform', 'speckle', 'periodic']
        
        # Ensure we have enough noise types for the number of classes
        if len(noise_types) < n_classes:
            # Repeat noise types if necessary
            noise_types = (noise_types * ((n_classes // len(noise_types)) + 1))[:n_classes]
        
        # Initialize images and labels with channel dimension
        X = np.zeros((n_samples, height, width, 1))
        y = np.zeros(n_samples, dtype=int)
        
        # Samples per class
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        sample_idx = 0
        for class_idx in range(n_classes):
            # Adjust for remainder
            n_samples_i = samples_per_class + (1 if class_idx < remainder else 0)
            
            # Get noise type for this class
            noise_type = noise_types[class_idx]
            
            for i in range(n_samples_i):
                # Generate noise
                if noise_type == 'gaussian':
                    img = np.random.normal(0.5, 0.2, (height, width))
                elif noise_type == 'salt_pepper':
                    img = np.random.choice([0, 1], size=(height, width), p=[0.9, 0.1])
                elif noise_type == 'uniform':
                    img = np.random.uniform(0, 1, (height, width))
                elif noise_type == 'speckle':
                    img = np.random.normal(0.5, 0.2, (height, width))
                    img += img * np.random.normal(0, 0.1, img.shape)
                elif noise_type == 'periodic':
                    y_indices, x_indices = np.ogrid[:height, :width]
                    frequency = np.random.uniform(0.1, 0.5)
                    img = 0.5 * (1 + np.sin(2 * np.pi * frequency * x_indices / width))
                
                # Add random variations
                img += np.random.normal(0, 0.1, (height, width))
                
                # Clip values to [0, 1]
                img = np.clip(img, 0, 1)
                
                # Store image and label with channel dimension
                X[sample_idx, :, :, 0] = img
                y[sample_idx] = class_idx
                sample_idx += 1
        
        return X, y
    
    def _generate_gradients(self, 
                           n_samples: int, 
                           n_classes: int,
                           image_size: Tuple[int, int] = (32, 32),
                           gradient_types: Optional[List[str]] = None,
                           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate images with different gradient patterns for each class.
        
        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes
            image_size: Size of the generated images (height, width)
            gradient_types: List of gradient types to use (default: ['linear', 'radial', 'angular'])
            **kwargs: Additional parameters
            
        Returns:
            Tuple containing images (X) and labels (y)
        """
        height, width = image_size
        
        # Default gradient types
        if gradient_types is None:
            gradient_types = ['linear', 'radial', 'angular', 'bilinear', 'spiral']
        
        # Ensure we have enough gradient types for the number of classes
        if len(gradient_types) < n_classes:
            # Repeat gradient types if necessary
            gradient_types = (gradient_types * ((n_classes // len(gradient_types)) + 1))[:n_classes]
        
        # Initialize images and labels with channel dimension
        X = np.zeros((n_samples, height, width, 1))
        y = np.zeros(n_samples, dtype=int)
        
        # Samples per class
        samples_per_class = n_samples // n_classes
        remainder = n_samples % n_classes
        
        sample_idx = 0
        for class_idx in range(n_classes):
            # Adjust for remainder
            n_samples_i = samples_per_class + (1 if class_idx < remainder else 0)
            
            # Get gradient type for this class
            gradient_type = gradient_types[class_idx]
            
            for i in range(n_samples_i):
                # Create base image
                img = np.zeros((height, width))
                
                # Generate coordinates
                y_indices, x_indices = np.ogrid[:height, :width]
                
                # Normalize coordinates to [0, 1]
                x_norm = x_indices / width
                y_norm = y_indices / height
                
                # Random parameters
                angle = np.random.uniform(0, 2 * np.pi)
                center_x = np.random.uniform(0.3, 0.7)
                center_y = np.random.uniform(0.3, 0.7)
                
                if gradient_type == 'linear':
                    # Linear gradient
                    # Direction vector
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    
                    # Project coordinates onto direction vector
                    projection = dx * x_norm + dy * y_norm
                    
                    # Normalize projection to [0, 1]
                    img = projection
                
                elif gradient_type == 'radial':
                    # Radial gradient
                    # Distance from center
                    dist = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2)
                    
                    # Normalize distance to [0, 1]
                    max_dist = np.sqrt(0.5**2 + 0.5**2)
                    img = 1 - (dist / max_dist)
                
                elif gradient_type == 'angular':
                    # Angular gradient
                    # Angle from center
                    dx = x_norm - center_x
                    dy = y_norm - center_y
                    angle_from_center = np.arctan2(dy, dx)
                    
                    # Normalize angle to [0, 1]
                    img = (angle_from_center + np.pi) / (2 * np.pi)
                
                elif gradient_type == 'bilinear':
                    # Bilinear gradient
                    img = x_norm * y_norm
                
                elif gradient_type == 'spiral':
                    # Spiral gradient
                    # Distance from center
                    dist = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2)
                    
                    # Angle from center
                    dx = x_norm - center_x
                    dy = y_norm - center_y
                    angle_from_center = np.arctan2(dy, dx)
                    
                    # Spiral function
                    spiral = (angle_from_center + dist * 10) % (2 * np.pi)
                    
                    # Normalize to [0, 1]
                    img = spiral / (2 * np.pi)
                
                # Add random variations
                img += np.random.normal(0, 0.1, (height, width))
                
                # Clip values to [0, 1]
                img = np.clip(img, 0, 1)
                
                # Store image and label with channel dimension
                X[sample_idx, :, :, 0] = img
                y[sample_idx] = class_idx
                sample_idx += 1
        
        return X, y

    def preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images by scaling pixel values to the range [0, 1].
        
        Args:
            images: Array of images to preprocess
            
        Returns:
            Preprocessed images
        """
        # Scale pixel values to the range [0, 1]
        images = images.astype('float32') / 255.0
        
        # If images don't have a channel dimension, add it
        if len(images.shape) == 3:
            images = images[..., np.newaxis]
            
        return images

    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   test_size: float = 0.2, 
                   val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            X: Array of images
            y: Array of labels
            test_size: Proportion of the data to include in the test split
            val_size: Proportion of the training data to include in the validation split
            
        Returns:
            Dictionary containing the training, validation, and test splits
        """
        # Split into training + validation and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        
        # Calculate validation size as a proportion of the training + validation set
        val_size_adjusted = val_size / (1 - test_size)
        
        # Split training + validation set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=self.random_state)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def visualize_images(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         n_images_per_class: int = 5, 
                         title: str = 'Generated Images', 
                         save_path: Optional[str] = None) -> None:
        """
        Visualize a sample of generated images.
        
        Args:
            X: Array of images
            y: Array of labels
            n_images_per_class: Number of images to display per class
            title: Title of the plot
            save_path: Path to save the plot (if None, the plot will be shown)
        """
        n_classes = len(np.unique(y))
        fig, axes = plt.subplots(n_classes, n_images_per_class, figsize=(n_images_per_class * 2, n_classes * 2))
        fig.suptitle(title, fontsize=16)
        
        # Ensure X has the right shape for visualization
        if len(X.shape) == 4:  # If X has a channel dimension
            X_vis = X.squeeze(axis=3)  # Remove channel dimension for visualization
        else:
            X_vis = X
        
        for class_idx in range(n_classes):
            class_images = X_vis[y == class_idx]
            for img_idx in range(min(n_images_per_class, len(class_images))):
                ax = axes[class_idx, img_idx]
                ax.imshow(class_images[img_idx], cmap='gray')
                ax.axis('off')
                if img_idx == 0:
                    ax.set_ylabel(f'Class {class_idx}', fontsize=12)
            # Hide any remaining empty subplots
            for img_idx in range(len(class_images), n_images_per_class):
                ax = axes[class_idx, img_idx]
                ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
