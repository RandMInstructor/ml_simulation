"""
Common Functions Module for for Running a Demo of the ML Simulation Environment.

This module provides functionality for demonstrating the package functionality of analyzing model residuals,
including confusion matrix generation, intermediate representation analysis,
cosine similarity calculation, and entropy analysis for image classification.
"""
import numpy as np
import os
import tensorflow as tf
from scipy.stats import entropy
from scipy.special import softmax

from src.training_pipeline import train_val_split
from src.model_selector import get_model_selector_cnn_example_model

from src.residuals_analyzer import ResidualsAnalyzer
from src.model_selector import ModelSelector
from src.training_pipeline import TrainingPipeline
from src.image_data_generator import ImageDataGenerator

from src.visualization_analyzer import VisualizationAnalyzer
from src.enhanced_visualization import DataVisualizer

from src.training_pipeline import example_training
from src.training_pipeline import train_from_config

def visualize_results(config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results):
    """Visualize results with statistical tests."""
    print("\n=== Visualizing Results with Statistical Tests ===")
    
    visualization_results_path = os.path.join(config['output_dir'], 'visualization')

    # Create visualization analyzer
    visualizer = VisualizationAnalyzer(save_dir=visualization_results_path)
    
    # Run full visualization
    visualization_results = visualizer.run_full_visualization(
        representations=intermediate_reps,
        softmax_outputs=softmax_outputs,
        y_true=data_splits['y_test'],
        y_pred=y_pred
    )
    
    print(f"Visualization results saved to: {visualization_results_path}")
    print(f"Statistical tests include both Pearson and Spearman correlation coefficients")
    
    return visualization_results

def enhanced_visualization(config : dict, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results):
    """Perform enhanced visualization for data inspection."""
    print("\n=== Performing Enhanced Visualization ===")
    enhanced_visualization_path = os.path.join(config['output_dir'], 'enhanced_visualization')
    class_names = ''
    if 'class_names' in config:
        class_names = config.get('class_names', None)

    # Create data visualizer
    visualizer = DataVisualizer(save_dir=enhanced_visualization_path)
    
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
        entropies=entropies,
        class_names=class_names
    )
    
    print(f"Enhanced visualization results saved to: {enhanced_visualization_path}")
    
    return visualization_paths



def analyze_residuals(config, pipeline : TrainingPipeline, data_splits, model_selector : ModelSelector):
    """Perform residuals analysis on the trained model."""
    print("\n=== Performing Residuals Analysis ===")
    
    # Create residuals analyzer
    analyzer = ResidualsAnalyzer(save_dir=config['output_dir'])

    # Get test data
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
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

    # Get predictions
    y_pred_pipeline = pipeline.predict(X_test)
    # Get predictions
    y_pred_softmax = np.argmax(softmax_outputs, axis=1)
    try:
        assert np.array_equal(y_pred_pipeline, y_pred_softmax)
    except:
        print("Predictions are not equal")
        print(f"{y_pred_pipeline[:10]}...")
        print(f"{y_pred_softmax[:10]}...")
        print("DONE: See what's going on here with the softmax outputs - Had something weird going on with the model")
    
    y_pred = y_pred_pipeline

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

# Define a function to save the generated data and a paired function to load saved data for future use. The functions should take config dict as a standard argument.
def save_generated_data(config, data_splits, class_names):
    """Save generated data to disk."""
    print("\n=== Saving Generated Data ===")
    config_seed = config['random_seed']

    # Save data splits
    np.savez_compressed(
        os.path.join(config['output_dir'], f'generated_data_{config['data_method']}_{config_seed}.npz'),
        X_train=data_splits['X_train'],
        y_train=data_splits['y_train'],
        X_val=data_splits['X_val'],
        y_val=data_splits['y_val'],
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test']
    )
    
    # Save class names
    with open(os.path.join(config['output_dir'], f'class_names_{config['data_method']}_{config_seed}.txt'), 'w') as file:
        for class_name in class_names:
            file.write(f"{class_name}\n")
    
    print(f"Generated data saved to: {config['output_dir']}")

def load_generated_data(config):
    """Load saved generated data."""
    print("\n=== Loading Saved Generated Data ===")
    
    config_seed = config['random_seed']

    # Load data splits
    data = np.load(os.path.join(config['output_dir'], f'generated_data_{config['data_method']}_{config_seed}.npz'))
    data_splits = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
        'X_test': data['X_test'],
        'y_test': data['y_test']
    }

    # Load class names
    with open(os.path.join(config['output_dir'], f'class_names_{config['data_method']}_{config_seed}.txt'), 'r') as file:
        class_names = [line.strip() for line in file]
    
    print(f"Saved data loaded from: {config['output_dir']}")
    
    return data_splits, class_names



def generate_data(config):
    """Generate synthetic image data."""
    print("\n=== Generating Synthetic Image Data ===")
    
    # Create data generator
    generator = ImageDataGenerator(random_state=config['random_seed'])
    
    # Generate images
    X, y, class_names= generator.generate_image_data(
        method=config['data_method'],
        n_samples=config['n_samples'],
        n_classes=config['n_classes'],
        image_size=config['image_size']
    )
    
    # Preprocess images
    X_processed = generator.preprocess_images(X)
    
    # Split data
    data_splits = generator.split_data(X_processed, y, test_size=0.1, val_size=0.1)
    
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
        class_names=class_names,
        save_path=os.path.join(config['output_dir'], 'sample_images.png')
    )
    
    return data_splits, class_names

def run_mnist_test():
    # Establish save path
    save_dir = './results/demo_components_main'
    os.makedirs(save_dir, exist_ok=True)

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    input_shape=(28, 28)

    # Create a separate validation set
    (x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train, val_size=0.1)
    data_splits = {
            'X_train': x_train,
            'y_train': y_train,
            'X_val': x_val,
            'y_val': y_val,
            'X_test': x_test,
            'y_test': y_test
        }

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create a model selector
    selector = ModelSelector(random_state=42)
    
    model = get_model_selector_cnn_example_model(input_shape=(28, 28, 1), selector=selector)
 
    # Create a training pipeline
    pipeline = TrainingPipeline(model=model, model_type='keras')
    try:
        pipeline.load_pipeline(save_dir)
    except:
        history = example_training(pipeline, x_train, y_train, x_val, y_val, None, None, save_dir=save_dir)

    pipeline.save_pipeline(save_dir)
    
    # Evaluate model
    metrics = pipeline.evaluate(
        X_test=data_splits['X_test'],
        y_test=data_splits['y_test']
    )
    
    config = {}
    config['output_dir'] = save_dir

    analyze_residuals(config, pipeline, {'X_test': x_test, 'y_test': y_test}, selector)

def run_generated_synthetic_test():
    # Establish save path
    save_dir = './results/demo_components_main/run_generated_synthetic_test'
    os.makedirs(save_dir, exist_ok=True)


    config = {}
    method = config['data_method'] = 'shapes'
    n_samples = config['n_samples'] = 10000
    n_classes=config['n_classes'] = 5
    image_size=config['image_size'] = (32, 32)
    random_seed = config['random_seed'] = 42
    config['output_dir'] = save_dir
    config['batch_size'] = 64
    config['epochs'] = 10
    config['early_stopping'] = True
    config['model_checkpoint'] = True
    config['enhanced_visualization'] = True
    
    try:
        data_splits, class_names = load_generated_data(config)
    except:
        data_splits, class_names = generate_data(config)
        save_generated_data(config, data_splits, class_names)

    config['class_names'] = class_names
    
    input_shape = image_size + (1,)

    x_train = data_splits['X_train']
    y_train = data_splits['y_train']
    x_val = data_splits['X_val']
    y_val = data_splits['y_val']
    x_test = data_splits['X_test']
    y_test = data_splits['y_test']

    # Preprocess the data
    #x_train = x_train.astype('float32') / 255.0
    #x_val = x_val.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0
    
    # Create a model selector
    selector = ModelSelector(random_state=random_seed)
    
    model = get_model_selector_cnn_example_model(input_shape=input_shape, selector=selector, config=config)
 
    # Create a training pipeline
    pipeline = TrainingPipeline(model=model, model_type='keras')
    try:
        pipeline.load_pipeline(save_dir)
    except:
        #history = example_training(pipeline, x_train, y_train, x_val, y_val, None, None, save_dir=save_dir)
        # Get callbacks
        history = train_from_config(config=config, pipeline=pipeline, data_splits=data_splits)

    pipeline.save_pipeline(save_dir)
    
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


    #analyze_residuals(config, pipeline, {'X_test': x_test, 'y_test': y_test}, selector)
    
    # Perform residuals analysis
    intermediate_reps, softmax_outputs, y_pred, residuals_results = analyze_residuals(
        config, pipeline, data_splits, selector
    )

    compare_entropies(intermediate_reps, softmax_outputs, temperature=1.0)
    
    # Visualize results with statistical tests
    visualization_results = visualize_results(
        config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results
    )
    
    # Perform enhanced visualization if requested
    if 'enhanced_visualization' in config:
        if config['enhanced_visualization']:
            visualization_paths = enhanced_visualization(
                config, data_splits, intermediate_reps, softmax_outputs, y_pred, residuals_results
            )

def calculate_entropy(vectors: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Calculate the entropy for each vector in the given set of vectors.
    
    Args:
        vectors: A 2D numpy array where each row is a vector.
        temperature: Temperature parameter to scale the logits before applying softmax.
        
    Returns:
        A 1D numpy array containing the entropy for each vector.
    """
    # Apply temperature scaling and softmax
    scaled_logits = vectors / temperature
    probabilities = softmax(scaled_logits, axis=1)
    
    # Calculate entropy for each vector
    entropies = entropy(probabilities, axis=1)
    
    return entropies

# Example usage with intermediate_reps and softmax_outputs
def compare_entropies(intermediate_reps: np.ndarray, softmax_outputs: np.ndarray, temperature: float = 1.0):
    """
    Compare the entropies of intermediate representations and softmax outputs.
    
    Args:
        intermediate_reps: A 2D numpy array of intermediate representations.
        softmax_outputs: A 2D numpy array of softmax outputs.
        temperature: Temperature parameter to scale the logits before applying softmax.
        
    Returns:
        A tuple containing the entropies of intermediate representations and softmax outputs.
    """
    # Calculate entropies
    intermediate_entropies = calculate_entropy(intermediate_reps, temperature)
    softmax_entropies = calculate_entropy(softmax_outputs, temperature)
    
    # Print comparison
    print("Intermediate Representations Entropies:")
    print(intermediate_entropies)
    print("\nSoftmax Outputs Entropies:")
    print(softmax_entropies)
    
    return intermediate_entropies, softmax_entropies

# Example call within the existing code
def example_entropy_comparison():
    # Assuming intermediate_reps and softmax_outputs are already defined
    intermediate_reps = np.random.rand(100, 10)  # Example data
    softmax_outputs = np.random.rand(100, 10)  # Example data
    
    # Compare entropies
    compare_entropies(intermediate_reps, softmax_outputs, temperature=1.0)

if __name__ == '__main__':
    example_entropy_comparison()
    #run_mnist_test()
    run_generated_synthetic_test()