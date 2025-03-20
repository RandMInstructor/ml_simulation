"""
Demonstration Script for ML Simulation Environment with Residuals Analysis.

This script demonstrates the complete workflow of the ML simulation environment
for multi-class image classification, including:
1. Generating synthetic image data
2. Selecting and configuring a machine learning model
3. Training and evaluating the model
4. Performing residuals analysis with intermediate representations
5. Visualizing results and statistical tests

The demonstration focuses on the relationship between cosine similarity of
intermediate representations and entropy from softmax outputs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
from typing import Dict, Any, Optional, List, Union, Tuple

# Import custom modules
from src.image_data_generator import ImageDataGenerator
from src.model_selector import ModelSelector
from src.training_pipeline import TrainingPipeline
from src.residuals_analyzer import ResidualsAnalyzer
from src.visualization_analyzer import VisualizationAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML Simulation Environment Demo')
    
    # Data generation parameters
    parser.add_argument('--data_method', type=str, default='shapes',
                        choices=['shapes', 'patterns', 'noise', 'gradients'],
                        help='Method for generating synthetic image data')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Size of the generated images (square)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['mlp', 'cnn', 'rnn'],
                        help='Type of model to use')
    parser.add_argument('--hidden_layers', type=str, default='128,64',
                        help='Comma-separated list of neurons in hidden layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment for reproducibility."""
    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up TensorFlow to use memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Logical GPUs: {logical_gpus}")
            print("-" * 50)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU: {gpu}")
                #Print GPU Details
                print(f"GPU: {gpu}")
                print(f"Name: {gpu.name}")
                print(f"Device Type: {gpu.device_type}")
                #print(f"Memory Limit: {gpu.memory_limit}")
                print("-" * 50)

                print()
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPUs detected")
    
    # Return configuration
    config = {
        'data_method': args.data_method,
        'n_samples': args.n_samples,
        'n_classes': args.n_classes,
        'image_size': (args.image_size, args.image_size),
        'model_type': args.model_type,
        'hidden_layers': [int(x) for x in args.hidden_layers.split(',')],
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'random_seed': args.random_seed
    }
    
    return config


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
        image_size=config['image_size']
    )
    
    # Preprocess images
    X_processed = generator.preprocess_images(X)
    
    # Split data
    data_splits = generator.split_data(X_processed, y, test_size=0.2, val_size=0.1)
    
    # Print data information
    print(f"Data method: {config['data_method']}")
    print(f"Number of classes: {config['n_classes']}")
    print(f"Image shape: {X.shape[1:]} (processed: {X_processed.shape[1:]})")
    print(f"Total samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Training samples: {data_splits['X_train'].shape[0]}")
    print(f"Validation samples: {data_splits['X_val'].shape[0]}")
    print(f"Test samples: {data_splits['X_test'].shape[0]}")
    
    # Visualize sample images
    generator.visualize_images(
        X, y,
        n_images_per_class=5,
        title=f'Generated Images - {config["data_method"].capitalize()} Method',
        save_path=os.path.join(config['output_dir'], 'sample_images.png')
    )
    
    return X, y, data_splits


def create_model(config, input_shape, n_classes):
    """Create and configure a machine learning model."""
    print("\n=== Creating Model ===")
    
    # Create model selector
    selector = ModelSelector(random_state=config['random_seed'])
    
    # Get model
    model = selector.get_model(
        model_type=config['model_type'],
        n_classes=n_classes,
        n_features=np.prod(input_shape),  # Flatten input shape
        hidden_layers=config['hidden_layers'],
        learning_rate=config['learning_rate']
    )
    
    # Print model information
    print(f"Model type: {config['model_type']}")
    print(f"Hidden layers: {config['hidden_layers']}")
    model.summary()
    
    return model, selector


def train_model(config, model, data_splits):
    """Train the model on the generated data."""
    print("\n=== Training Model ===")
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        model=model,
        model_type='keras',
        save_dir=config['output_dir']
    )
    
    # Get callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['output_dir'], 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = pipeline.train(
        X_train=data_splits['X_train'],
        y_train=data_splits['y_train'],
        X_val=data_splits['X_val'],
        y_val=data_splits['y_val'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=callbacks
    )
    
    # Evaluate model
    metrics = pipeline.evaluate(
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test']
    )
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training history
    pipeline.plot_training_history(
        save_path=os.path.join(config['output_dir'], 'training_history.png')
    )
    
    # Plot confusion matrix
    pipeline.plot_confusion_matrix(
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test'],
        normalize='true',
        save_path=os.path.join(config['output_dir'], 'confusion_matrix.png')
    )
    
    return pipeline, history, metrics


def analyze_residuals(config, pipeline, model_selector, data_splits):
    """Perform residuals analysis on the trained model."""
    print("\n=== Performing Residuals Analysis ===")
    
    # Get test data
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Extract intermediate representations
    intermediate_reps = pipeline.extract_intermediate_representations(
        X_test,
        intermediate_model=model_selector.intermediate_representations['model']
    )
    
    # Extract softmax outputs
    softmax_outputs = pipeline.extract_softmax_outputs(
        X_test,
        softmax_model=model_selector.softmax_outputs['model']
    )
    
    # Create residuals analyzer
    analyzer = ResidualsAnalyzer(save_dir=config['output_dir'])
    
    # Run full analysis
    residuals_results = analyzer.run_full_analysis(
        representations=intermediate_reps,
        softmax_outputs=softmax_outputs,
        y_true=y_test,
        y_pred=y_pred,
        save_dir=os.path.join(config['output_dir'], 'residuals')
    )
    
    # Print analysis information
    print(f"Intermediate representations shape: {intermediate_reps.shape}")
    print(f"Softmax outputs shape: {softmax_outputs.shape}")
    print(f"Number of correctly classified samples: {np.sum(y_test == y_pred)}")
    print(f"Number of misclassified samples: {np.sum(y_test != y_pred)}")
    
    return intermediate_reps, softmax_outputs, y_pred, residuals_results


def visualize_results(config, intermediate_reps, softmax_outputs, y_test, y_pred):
    """Visualize the results with advanced statistical tests."""
    print("\n=== Visualizing Results with Statistical Tests ===")
    
    # Create visualization analyzer
    visualizer = VisualizationAnalyzer(
        save_dir=os.path.join(config['output_dir'], 'visualization')
    )
    
    # Run full visualization
    visualization_results = visualizer.run_full_visualization(
        representations=intermediate_reps,
        softmax_outputs=softmax_outputs,
        y_true=y_test,
        y_pred=y_pred
    )
    
    # Print visualization information
    print(f"Visualization results saved to: {os.path.join(config['output_dir'], 'visualization')}")
    print("Statistical tests include both Pearson and Spearman correlation coefficients")
    
    return visualization_results


def main():
    """Main function to run the demonstration."""

    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    config = setup_environment(args)
    
    # Generate data
    X, y, data_splits = generate_data(config)
    
    # Create model
    model, model_selector = create_model(
        config,
        input_shape=data_splits['X_train'].shape[1:],
        n_classes=len(np.unique(y))
    )
    
    # Train model
    pipeline, history, metrics = train_model(config, model, data_splits)
    
    # Analyze residuals
    intermediate_reps, softmax_outputs, y_pred, residuals_results = analyze_residuals(
        config, pipeline, model_selector, data_splits
    )
    
    # Visualize results
    visualization_results = visualize_results(
        config, intermediate_reps, softmax_outputs, data_splits['y_test'], y_pred
    )
    
    print("\n=== Demonstration Complete ===")
    print(f"All results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
