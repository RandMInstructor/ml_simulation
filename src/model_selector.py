"""
Model Selection Module for ML Simulation Environment.

This module provides customizable model architectures and hyperparameters
for multi-class classification problems.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Optional, List, Union, Tuple, Callable


class ModelSelector:
    """
    A class for selecting and configuring machine learning models for classification.
    
    This class provides various model architectures with customizable hyperparameters
    for multi-class classification problems.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelSelector.
        
        Args:
            random_state: Seed for random number generation for reproducibility
        """
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Dictionary to store intermediate representations
        self.intermediate_representations = {}
        
        # Dictionary to store softmax outputs
        self.softmax_outputs = {}
    
    def get_model(self, 
                 model_type: str = 'mlp', 
                 n_classes: int = 3, 
                 n_features: int = 2,
                 **kwargs) -> Any:
        """
        Get a model of the specified type with the given hyperparameters.
        
        Args:
            model_type: Type of model to create
                       ('mlp', 'cnn', 'rnn', 'random_forest', 'gradient_boosting', 'svm', 'logistic')
            n_classes: Number of classes for classification
            n_features: Number of input features
            **kwargs: Additional hyperparameters specific to the chosen model
            
        Returns:
            The configured model
        """
        if model_type == 'mlp':
            return self._get_mlp_model(n_classes, n_features, **kwargs)
        elif model_type == 'cnn':
            return self._get_cnn_model(n_classes, n_features, **kwargs)
        elif model_type == 'rnn':
            return self._get_rnn_model(n_classes, n_features, **kwargs)
        elif model_type == 'random_forest':
            return self._get_random_forest_model(n_classes, **kwargs)
        elif model_type == 'gradient_boosting':
            return self._get_gradient_boosting_model(n_classes, **kwargs)
        elif model_type == 'svm':
            return self._get_svm_model(n_classes, **kwargs)
        elif model_type == 'logistic':
            return self._get_logistic_regression_model(n_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_mlp_model(self, 
                      n_classes: int, 
                      n_features: int,
                      hidden_layers: List[int] = [128, 64],
                      activation: str = 'relu',
                      dropout_rate: float = 0.2,
                      l2_reg: float = 0.001,
                      learning_rate: float = 0.001,
                      **kwargs) -> tf.keras.Model:
        """
        Create a Multi-Layer Perceptron (MLP) model.
        
        Args:
            n_classes: Number of classes for classification
            n_features: Number of input features
            hidden_layers: List of neurons in each hidden layer
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for optimizer
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured MLP model
        """
        # Input layer
        inputs = layers.Input(shape=(n_features,), name='input')
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(
                units=units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
            x = layers.Dropout(rate=dropout_rate, name=f'dropout_{i}')(x)
        
        # Intermediate representation layer (before final classification)
        intermediate = layers.Dense(
            units=32,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='intermediate_representation'
        )(x)
        
        # Output layer
        outputs = layers.Dense(
            units=n_classes,
            activation='softmax',
            name='output'
        )(intermediate)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create a separate model to extract intermediate representations
        intermediate_model = models.Model(inputs=inputs, outputs=intermediate)
        
        # Store the intermediate model
        self.intermediate_representations['model'] = intermediate_model
        
        # Create a separate model to extract softmax outputs
        softmax_model = models.Model(inputs=inputs, outputs=outputs)
        
        # Store the softmax model
        self.softmax_outputs['model'] = softmax_model
        
        return model
    
    def _get_cnn_model(self, 
                      n_classes: int, 
                      n_features: int,
                      conv_layers: List[int] = [32, 64, 128],
                      kernel_size: int = 3,
                      pool_size: int = 2,
                      dense_layers: List[int] = [128],
                      activation: str = 'relu',
                      dropout_rate: float = 0.2,
                      l2_reg: float = 0.001,
                      learning_rate: float = 0.001,
                      **kwargs) -> tf.keras.Model:
        """
        Create a Convolutional Neural Network (CNN) model.
        
        Args:
            n_classes: Number of classes for classification
            n_features: Number of input features
            conv_layers: List of filters in each convolutional layer
            kernel_size: Size of convolutional kernels
            pool_size: Size of pooling windows
            dense_layers: List of neurons in each dense layer
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for optimizer
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured CNN model
        """
        # For image data, we expect the input to already be in the correct shape (height, width, channels)
        # Instead of trying to reshape, we'll use the input shape directly
        
        # Input layer - directly use the shape from the data
        inputs = layers.Input(shape=(32, 32, 1), name='input')
        
        # Convolutional layers
        x = inputs
        for i, filters in enumerate(conv_layers):
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                padding='same',
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'conv_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_conv_{i}')(x)
            x = layers.MaxPooling2D(pool_size=pool_size, name=f'pool_{i}')(x)
            x = layers.Dropout(rate=dropout_rate, name=f'dropout_conv_{i}')(x)
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers
        for i, units in enumerate(dense_layers):
            x = layers.Dense(
                units=units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_dense_{i}')(x)
            x = layers.Dropout(rate=dropout_rate, name=f'dropout_dense_{i}')(x)
        
        # Intermediate representation layer (before final classification)
        intermediate = layers.Dense(
            units=32,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='intermediate_representation'
        )(x)
        
        # Output layer
        outputs = layers.Dense(
            units=n_classes,
            activation='softmax',
            name='output'
        )(intermediate)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create a separate model to extract intermediate representations
        intermediate_model = models.Model(inputs=inputs, outputs=intermediate)
        
        # Store the intermediate model
        self.intermediate_representations['model'] = intermediate_model
        
        # Create a separate model to extract softmax outputs
        softmax_model = models.Model(inputs=inputs, outputs=outputs)
        
        # Store the softmax model
        self.softmax_outputs['model'] = softmax_model
        
        return model
    
    def _get_rnn_model(self, 
                      n_classes: int, 
                      n_features: int,
                      rnn_units: List[int] = [64, 32],
                      rnn_type: str = 'lstm',
                      bidirectional: bool = True,
                      dense_layers: List[int] = [64],
                      activation: str = 'relu',
                      dropout_rate: float = 0.2,
                      l2_reg: float = 0.001,
                      learning_rate: float = 0.001,
                      **kwargs) -> tf.keras.Model:
        """
        Create a Recurrent Neural Network (RNN) model.
        
        Args:
            n_classes: Number of classes for classification
            n_features: Number of input features
            rnn_units: List of units in each RNN layer
            rnn_type: Type of RNN cell ('lstm', 'gru', 'simple')
            bidirectional: Whether to use bidirectional RNN
            dense_layers: List of neurons in each dense layer
            activation: Activation function for dense layers
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for optimizer
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured RNN model
        """
        # Reshape input for RNN
        sequence_length = int(np.sqrt(n_features))
        feature_dim = int(n_features / sequence_length)
        
        # Input layer
        inputs = layers.Input(shape=(sequence_length, feature_dim), name='input')
        
        # RNN layers
        x = inputs
        for i, units in enumerate(rnn_units):
            # Select RNN type
            if rnn_type == 'lstm':
                rnn_layer = layers.LSTM
            elif rnn_type == 'gru':
                rnn_layer = layers.GRU
            else:  # simple
                rnn_layer = layers.SimpleRNN
            
            # Add RNN layer
            return_sequences = i < len(rnn_units) - 1  # Return sequences for all but the last RNN layer
            
            if bidirectional:
                x = layers.Bidirectional(
                    rnn_layer(
                        units=units,
                        return_sequences=return_sequences,
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=f'rnn_{i}'
                    )
                )(x)
            else:
                x = rnn_layer(
                    units=units,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=f'rnn_{i}'
                )(x)
            
            x = layers.BatchNormalization(name=f'batch_norm_rnn_{i}')(x)
            x = layers.Dropout(rate=dropout_rate, name=f'dropout_rnn_{i}')(x)
        
        # Dense layers
        for i, units in enumerate(dense_layers):
            x = layers.Dense(
                units=units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_dense_{i}')(x)
            x = layers.Dropout(rate=dropout_rate, name=f'dropout_dense_{i}')(x)
        
        # Intermediate representation layer (before final classification)
        intermediate = layers.Dense(
            units=32,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
            name='intermediate_representation'
        )(x)
        
        # Output layer
        outputs = layers.Dense(
            units=n_classes,
            activation='softmax',
            name='output'
        )(intermediate)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create a separate model to extract intermediate representations
        intermediate_model = models.Model(inputs=inputs, outputs=intermediate)
        
        # Store the intermediate model
        self.intermediate_representations['model'] = intermediate_model
        
        # Create a separate model to extract softmax outputs
        softmax_model = models.Model(inputs=inputs, outputs=outputs)
        
        # Store the softmax model
        self.softmax_outputs['model'] = softmax_model
        
        return model
    
    def _get_random_forest_model(self, 
                               n_classes: int,
                               n_estimators: int = 100,
                               max_depth: Optional[int] = None,
                               min_samples_split: int = 2,
                               min_samples_leaf: int = 1,
                               **kwargs) -> RandomForestClassifier:
        """
        Create a Random Forest Classifier.
        
        Args:
            n_classes: Number of classes for classification
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured Random Forest Classifier
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            **kwargs
        )
        
        return model
    
    def _get_gradient_boosting_model(self, 
                                   n_classes: int,
                                   n_estimators: int = 100,
                                   learning_rate: float = 0.1,
                                   max_depth: int = 3,
                                   min_samples_split: int = 2,
                                   min_samples_leaf: int = 1,
                                   **kwargs) -> GradientBoostingClassifier:
        """
        Create a Gradient Boosting Classifier.
        
        Args:
            n_classes: Number of classes for classification
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured Gradient Boosting Classifier
        """
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            **kwargs
        )
        
        return model
    
    def _get_svm_model(self, 
                     n_classes: int,
                     C: float = 1.0,
                     kernel: str = 'rbf',
                     gamma: str = 'scale',
                     **kwargs) -> SVC:
        """
        Create a Support Vector Machine Classifier.
        
        Args:
            n_classes: Number of classes for classification
            C: Regularization parameter
            kernel: Kernel type
            gamma: Kernel coefficient
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured SVM Classifier
        """
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=self.random_state,
            **kwargs
        )
        
        return model
    
    def _get_logistic_regression_model(self, 
                                     n_classes: int,
                                     C: float = 1.0,
                                     penalty: str = 'l2',
                                     solver: str = 'lbfgs',
                                     max_iter: int = 1000,
                                     **kwargs) -> LogisticRegression:
        """
        Create a Logistic Regression Classifier.
        
        Args:
            n_classes: Number of classes for classification
            C: Inverse of regularization strength
            penalty: Penalty norm
            solver: Algorithm for optimization
            max_iter: Maximum number of iterations
            **kwargs: Additional hyperparameters
            
        Returns:
            Configured Logistic Regression Classifier
        """
        # For multi-class problems, use 'multinomial' with 'lbfgs' solver
        if n_classes > 2 and solver == 'lbfgs':
            multi_class = 'multinomial'
        else:
            multi_class = 'auto'
        
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=self.random_state,
            **kwargs
        )
        
        return model


# Example usage
if __name__ == "__main__":
    # Create a model selector
    selector = ModelSelector(random_state=42)
    
    # Get a CNN model
    model = selector.get_model(
        model_type='cnn',
        n_classes=10,
        n_features=32*32,
        conv_layers=[32, 64, 128],
        dense_layers=[128, 64]
    )
    
    # Print model summary
    model.summary()
    
    # Get a Random Forest model
    rf_model = selector.get_model(
        model_type='random_forest',
        n_classes=10,
        n_estimators=100,
        max_depth=10
    )
    
    print(f"Random Forest model: {rf_model}")
