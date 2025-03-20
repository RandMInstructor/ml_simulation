"""
Updated Demo Script with Enhanced Visualization and Resume Training Functionality.

This script demonstrates the ML simulation environment for multi-class classification
with residuals analysis, enhanced visualization, and resume training capabilities.
"""

import argparse
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

# Import modules
from src.image_data_generator import ImageDataGenerator
from src.model_selector import ModelSelector
from src.training_pipeline import TrainingPipeline
from src.residuals_analyzer import ResidualsAnalyzer
from src.visualization_analyzer import VisualizationAnalyzer
from src.enhanced_visualization import DataVisualizer
from src.training_utilities import TrainingUtilities


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML Simulation Environment Demo')
    
    # Data generation parameters
    parser.add_argument('--data_method', type=str, default='shapes',
                      choices=['shapes', 'patterns', 'noise', 'gradients'],
                      help='Method for generating synthetic image data')
    parser.add_argument('--n_samples', type=int, default=500,
                      help='Number of samples to generate')
    parser.add_argument('--n_classes', type=int, default=5,
                      help='Number of classes')
    parser.add_argument('--image_size', type=int, default=32,
                      help='Size of generated images (width and height)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn',
                      choices=['mlp', 'cnn', 'rnn', 'random_forest', 'svm'],
                      help='Type of model to use')
    parser.add_argument('--hidden_layers', type=str, default='128,64',
                      help='Comma-separated list of hidden layer sizes')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for training')
    
    # Random seed
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Results directory
    parser.add_argument('--results_dir', type=str, default='./results',
                      help='Directory to save results')
    
    # Enhanced visualization options
    parser.add_argument('--enhanced_visualization', action='store_true',
                      help='Enable enhanced visualization')
    
    # Resume training options
    parser.add_argument('--resume_training', action='store_true',
                      help='Resume training from a saved model')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to saved model for resuming training')
    parser.add_argument('--additional_epochs', type=int, default=10,
                      help='Number of additional epochs when resuming training')
    
    return parser.parse_args()


def generate_data(config):
    """Generate synthetic image data."""
    print("\n=== Generating Synthetic Image Data ===")
    
    # Create data generator
    generator = ImageDataGenerator(random_state=config['random_seed'])
    
    # Generate images
    X, y = generator.generate_image_data(
        method=config['data_method'],
        n_samples=config['n_samples'],
        n_classes=config['n_classes'],
        image_size=(config['image_size'], config['image_size'])
    )
    
    # Preprocess images
    X_processed = generator.preprocess_images(X)
    
    # Split data
    data_splits = generator.split_data(X_processed, y, test_size=0.2, val_size=0.1)
    
    return data_splits


def create_model(config, input_shape, n_classes):
    """Create a machine learning model."""
    print("\n=== Creating Model ===")
    
    # Parse hidden layers
    hidden_layers = [int(size) for size in config['hidden_layers'].split(',')]
    
    # Create model selector
    selector = ModelSelector()
    
    # Create model
    if config['model_type'] in ['mlp', 'cnn', 'rnn']:
        # Create Keras model
        model = selector.get_model(
            model_type=config['model_type'],
            n_classes=n_classes,
            n_features=input_shape[0] if len(input_shape) == 1 else input_shape,
            hidden_layers=[int(size) for size in config['hidden_layers'].split(',')],
            learning_rate=config['learning_rate']
        )
        model_type = 'keras'
    else:
        # Create scikit-learn model
        model = selector.get_model(
            model_type=config['model_type'],
            n_classes=n_classes,
            n_features=input_shape[0] if len(input_shape) == 1 else input_shape[0] * input_shape[1],
            random_state=config['random_seed']
        )
        model_type = 'sklearn'
    
    return model, model_type, selector


def train_model(config, model, model_type, data_splits):
    """Train the model."""
    print("\n=== Training Model ===")
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        model=model,
        model_type=model_type,
        save_dir=config['results_dir']
    )
    
    # Train the model
    if model_type == 'keras':
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(config['results_dir'], 'best_model.h5'),
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
            X_train=data_splits['X_train'],
            y_train=data_splits['y_train'],
            X_val=data_splits['X_val'],
            y_val=data_splits['y_val'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            callbacks=callbacks
        )
    else:
        # Train without validation
        history = pipeline.train(
            X_train=data_splits['X_train'],
            y_train=data_splits['y_train']
        )
    
    # Evaluate the model
    metrics = pipeline.evaluate(
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test']
    )
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model
    model_path = os.path.join(config['results_dir'], 'final_model.h5')
    if model_type == 'keras':
        model.save(model_path)
    else:
        import joblib
        joblib.dump(model, model_path)
    
    # Save training history
    import json
    with open(os.path.join(config['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return pipeline, metrics


def resume_training(config, data_splits):
    """Resume training from a saved model."""
    print("\n=== Resuming Training ===")
    
    # Create training utilities
    training_utils = TrainingUtilities(save_dir=config['results_dir'])
    
    # Determine model path
    model_path = config['model_path']
    if model_path is None:
        # Find the most recent model
        model_paths = training_utils.find_saved_models(config['results_dir'])
        if not model_paths:
            raise ValueError("No saved models found in the results directory.")
        model_path = model_paths[0]  # Use the first model found
    
    # Resume training
    model, history = training_utils.resume_training(
        model_path=model_path,
        X_train=data_splits['X_train'],
        y_train=data_splits['y_train'],
        X_val=data_splits['X_val'],
        y_val=data_splits['y_val'],
        additional_epochs=config['additional_epochs'],
        batch_size=config['batch_size']
    )
    
    # Plot training history
    history_path = os.path.join(config['results_dir'], 'training_history.png')
    training_utils.plot_training_history(
        history=history,
        save_path=history_path
    )
    
    # Create training pipeline for evaluation
    pipeline = TrainingPipeline(
        model=model,
        model_type='keras',  # Resumed models are always Keras models
        save_dir=config['results_dir']
    )
    
    # Evaluate the model
    metrics = pipeline.evaluate(
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test']
    )
    
    # Print evaluation metrics
    print("\nEvaluation Metrics After Resuming Training:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return pipeline, metrics


def analyze_residuals(config, pipeline, data_splits, selector=None):
    """Perform residuals analysis."""
    print("\n=== Performing Residuals Analysis ===")
    
    # Create residuals analyzer
    analyzer = ResidualsAnalyzer(save_dir=config['results_dir'])
    
    # Extract intermediate representations
    if selector is not None:
        # Use the selector's intermediate model
        intermediate_reps = pipeline.extract_intermediate_representations(
            X=data_splits['X_test'],
            intermediate_model=selector.intermediate_representations['model']
        )
        
        # Extract softmax outputs
        softmax_outputs = pipeline.extract_softmax_outputs(
            X=data_splits['X_test'],
            softmax_model=selector.softmax_outputs['model']
        )
    else:
        # For backward compatibility or when selector is not available
        print("Warning: Model selector not provided. Using direct prediction instead.")
        # Use direct prediction
        softmax_outputs = pipeline.predict_proba(X=data_splits['X_test'])
        # For intermediate representations, we'll use the last hidden layer output
        # This is a fallback and may not work as expected
        intermediate_reps = np.random.rand(len(data_splits['X_test']), 32)  # Placeholder
    
    # Get predictions
    y_pred = np.argmax(softmax_outputs, axis=1)
    
    # Run full residuals analysis
    residuals_results = analyzer.run_full_analysis(
        representations=intermediate_reps,
        softmax_outputs=softmax_outputs,
        y_true=data_splits['y_test'],
        y_pred=y_pred,
        save_dir=os.path.join(config['results_dir'], 'residuals')
    )
    
    # Print some information about the residuals analysis
    print(f"Intermediate representations shape: {intermediate_reps.shape}")
    print(f"Softmax outputs shape: {softmax_outputs.shape}")
    print(f"Number of correctly classified samples: {np.sum(y_pred == data_splits['y_test'])}")
    print(f"Number of misclassified samples: {np.sum(y_pred != data_splits['y_test'])}")
    
    return intermediate_reps, softmax_outputs, y_pred, residuals_results


def visualize_results(config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results):
    """Visualize results with statistical tests."""
    print("\n=== Visualizing Results with Statistical Tests ===")
    
    # Create visualization analyzer
    visualizer = VisualizationAnalyzer(save_dir=os.path.join(config['results_dir'], 'visualization'))
    
    # Run full visualization
    visualization_results = visualizer.run_full_visualization(
        representations=intermediate_reps,
        softmax_outputs=softmax_outputs,
        y_true=data_splits['y_test'],
        y_pred=y_pred
    )
    
    print(f"Visualization results saved to: {os.path.join(config['results_dir'], 'visualization')}")
    print(f"Statistical tests include both Pearson and Spearman correlation coefficients")
    
    return visualization_results


def enhanced_visualization(config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results):
    """Perform enhanced visualization for data inspection."""
    print("\n=== Performing Enhanced Visualization ===")
    
    # Create data visualizer
    visualizer = DataVisualizer(save_dir=os.path.join(config['results_dir'], 'enhanced_visualization'))
    
    # Extract similarity matrices from residuals results
    similarity_matrices = residuals_results.get('cosine_similarity', {})
    
    # Extract entropies from residuals results
    entropies = residuals_results.get('entropy', np.array([]))
    
    # Run comprehensive visualization
    visualization_paths = visualizer.run_comprehensive_visualization(
        X_train=data_splits['X_train'],
        y_train=data_splits['y_train'],
        X_val=data_splits['X_val'],
        y_val=data_splits['y_val'],
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test'],
        y_pred=y_pred,
        intermediate_reps=intermediate_reps,
        softmax_outputs=softmax_outputs,
        similarity_matrices=similarity_matrices,
        entropies=entropies
    )
    
    print(f"Enhanced visualization results saved to: {os.path.join(config['results_dir'], 'enhanced_visualization')}")
    
    return visualization_paths


def main(args=None):
    """Main function."""
    # Parse arguments
    if args is None:
        args = parse_args()
    
    # Create configuration dictionary
    config = vars(args)
    
    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Set random seed
    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])
    
    # Generate data
    data_splits = generate_data(config)
    
    # Determine input shape
    if config['model_type'] == 'cnn':
        # For CNN, use the full image shape with channel dimension
        input_shape = (config['image_size'], config['image_size'], 1)
    else:
        # For other models, flatten the image
        input_shape = (config['image_size'] * config['image_size'],)
    
    # Training pipeline
    if config['resume_training']:
        # Resume training from a saved model
        pipeline, metrics = resume_training(config, data_splits)
    else:
        # Create and train a new model
        model, model_type, selector = create_model(config, input_shape, config['n_classes'])
        pipeline, metrics = train_model(config, model, model_type, data_splits)
    
    # Perform residuals analysis
    intermediate_reps, softmax_outputs, y_pred, residuals_results = analyze_residuals(
        config, pipeline, data_splits, selector
    )
    
    # Visualize results with statistical tests
    visualization_results = visualize_results(
        config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results
    )
    
    # Perform enhanced visualization if requested
    if config['enhanced_visualization']:
        visualization_paths = enhanced_visualization(
            config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results
        )
    
    print("\n=== Demonstration Complete ===")
    print(f"All results saved to: {config['results_dir']}")


if __name__ == '__main__':
    #main()
    #Here we will run the demo script as if the following command were executed in the terminal:
    #enhanced_demo.py --data_method shapes --n_classes 5 --model_type cnn --hidden_layers 128,64 --epochs 50 --n_samples 100 --enhanced_visualization
    #We will use the default values for the rest of the arguments
    args = parse_args()
    args.data_method = 'shapes'
    args.n_classes = 5
    args.model_type = 'cnn'
    args.hidden_layers = '128,64'
    args.epochs = 500
    args.n_samples = 100
    args.enhanced_visualization = True
    main(args)
