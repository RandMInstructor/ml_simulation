"""
Common Functions Module for for Running a Demo of the ML Simulation Environment.

This module provides functionality for demonstrating the package functionality of analyzing model residuals,
including confusion matrix generation, intermediate representation analysis,
cosine similarity calculation, and entropy analysis for image classification.
"""
import numpy as np
import os
import tensorflow as tf

from training_pipeline import train_val_split
from model_selector import generate_cnn_params

from training_pipeline import get_model_selector_cnn

from residuals_analyzer import ResidualsAnalyzer
from model_selector import ModelSelector
from training_pipeline import TrainingPipeline

def analyze_residuals(config, pipeline : TrainingPipeline, data_splits, model_selector : ModelSelector):
    """Perform residuals analysis on the trained model."""
    print("\n=== Performing Residuals Analysis ===")
    
    # Create residuals analyzer
    analyzer = ResidualsAnalyzer(save_dir=config['output_dir'])

    # Get test data
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    # Get predictions
    y_pred_pipeline = pipeline.predict(X_test)
    # Get predictions
    y_pred_softmax = np.argmax(softmax_outputs, axis=1)
    assert np.array_equal(y_pred_pipeline, y_pred_softmax)
    y_pred = y_pred_pipeline
    
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


if __name__ == '__main__':
    # Establish save path
    save_dir = './results/demo_components_main'

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    input_shape=(28, 28)

    # Create a separate validation set
    (x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train, val_size=0.1)

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create a model selector
    selector = ModelSelector(random_state=42)
    
    model = get_model_selector_cnn(input_shape=(28, 28, 1), selector=selector)
 

    
    # Print model summary
    model.summary()

    # Create a training pipeline
    pipeline = TrainingPipeline(model=model, model_type='keras')

    # Extract intermediate representations
    intermediate_reps = pipeline.extract_intermediate_representations(
        x_test,
        intermediate_model=selector.intermediate_representations['model']
    )
    # Extract softmax outputs
    softmax_outputs = pipeline.extract_softmax_outputs(
        x_test,
        softmax_model=selector.softmax_outputs['model']
    )