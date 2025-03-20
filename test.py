"""
Test script for ML Simulation Environment.

This script tests the main components of the ML simulation environment
to ensure they work correctly together.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Import custom modules
from src.image_data_generator import ImageDataGenerator
from src.model_selector import ModelSelector
from src.training_pipeline import TrainingPipeline
from src.residuals_analyzer import ResidualsAnalyzer
from src.visualization_analyzer import VisualizationAnalyzer


def test_image_data_generator():
    """Test the ImageDataGenerator class."""
    print("\n=== Testing ImageDataGenerator ===")
    
    # Create data generator
    generator = ImageDataGenerator(random_state=42)
    
    # Test all data generation methods
    methods = ['shapes', 'patterns', 'noise', 'gradients']
    
    for method in methods:
        print(f"\nTesting {method} method...")
        
        # Generate images
        X, y = generator.generate_image_data(
            method=method,
            n_samples=100,
            n_classes=3,
            image_size=(32, 32)
        )
        
        # Preprocess images
        X_processed = generator.preprocess_images(X)
        
        # Split data
        data_splits = generator.split_data(X_processed, y, test_size=0.2, val_size=0.1)
        
        # Check shapes and types
        assert X.shape == (100, 32, 32), f"Expected X shape (100, 32, 32), got {X.shape}"
        assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
        assert X_processed.shape == (100, 32, 32, 1), f"Expected X_processed shape (100, 32, 32, 1), got {X_processed.shape}"
        assert np.unique(y).size <= 3, f"Expected at most 3 unique classes, got {np.unique(y).size}"
        
        # Check data splits
        assert 'X_train' in data_splits, "X_train missing from data splits"
        assert 'y_train' in data_splits, "y_train missing from data splits"
        assert 'X_val' in data_splits, "X_val missing from data splits"
        assert 'y_val' in data_splits, "y_val missing from data splits"
        assert 'X_test' in data_splits, "X_test missing from data splits"
        assert 'y_test' in data_splits, "y_test missing from data splits"
        
        print(f"  ✓ {method} method passed all tests")
    
    print("\n✓ ImageDataGenerator tests passed")
    return True


def test_model_selector():
    """Test the ModelSelector class."""
    print("\n=== Testing ModelSelector ===")
    
    # Create model selector
    selector = ModelSelector(random_state=42)
    
    # Test deep learning models
    dl_models = ['mlp', 'cnn', 'rnn']
    
    for model_type in dl_models:
        print(f"\nTesting {model_type} model...")
        
        # Get model
        model = selector.get_model(
            model_type=model_type,
            n_classes=3,
            n_features=32*32,
            hidden_layers=[64, 32]
        )
        
        # Check model type
        assert isinstance(model, tf.keras.Model), f"Expected tf.keras.Model, got {type(model)}"
        
        # Check intermediate representations model
        assert 'model' in selector.intermediate_representations, "Intermediate representations model missing"
        assert isinstance(selector.intermediate_representations['model'], tf.keras.Model), \
            f"Expected intermediate_representations['model'] to be tf.keras.Model, got {type(selector.intermediate_representations['model'])}"
        
        # Check softmax outputs model
        assert 'model' in selector.softmax_outputs, "Softmax outputs model missing"
        assert isinstance(selector.softmax_outputs['model'], tf.keras.Model), \
            f"Expected softmax_outputs['model'] to be tf.keras.Model, got {type(selector.softmax_outputs['model'])}"
        
        print(f"  ✓ {model_type} model passed all tests")
    
    # Test traditional ML models
    ml_models = ['random_forest', 'gradient_boosting', 'svm', 'logistic']
    
    for model_type in ml_models:
        print(f"\nTesting {model_type} model...")
        
        # Get model
        model = selector.get_model(
            model_type=model_type,
            n_classes=3,
            n_features=32*32
        )
        
        # Check model type (should not be tf.keras.Model)
        assert not isinstance(model, tf.keras.Model), f"Expected non-tf.keras.Model, got {type(model)}"
        
        print(f"  ✓ {model_type} model passed all tests")
    
    print("\n✓ ModelSelector tests passed")
    return True


def test_training_pipeline():
    """Test the TrainingPipeline class with a small dataset."""
    print("\n=== Testing TrainingPipeline ===")
    
    # Create a small dataset
    n_samples = 100
    n_features = 32*32
    n_classes = 3
    
    X = np.random.randn(n_samples, 32, 32, 1)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    # Create a simple CNN model
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu', name='intermediate_representation'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create intermediate model
    intermediate_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer('intermediate_representation').output
    )
    
    # Create a model selector with the models
    selector = ModelSelector(random_state=42)
    selector.intermediate_representations['model'] = intermediate_model
    selector.softmax_outputs['model'] = model
    
    # Create training pipeline
    pipeline = TrainingPipeline(model=model, model_type='keras', save_dir='./test_results')
    
    # Test training
    print("\nTesting training...")
    history = pipeline.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=32,
        epochs=2  # Just a quick test
    )
    
    # Check history
    assert 'loss' in history, "Loss missing from history"
    assert 'accuracy' in history, "Accuracy missing from history"
    assert 'val_loss' in history, "Validation loss missing from history"
    assert 'val_accuracy' in history, "Validation accuracy missing from history"
    
    # Test evaluation
    print("\nTesting evaluation...")
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Check metrics
    assert 'loss' in metrics, "Loss missing from metrics"
    assert 'accuracy' in metrics, "Accuracy missing from metrics"
    
    # Test prediction
    print("\nTesting prediction...")
    y_pred = pipeline.predict(X_test)
    
    # Check predictions
    assert y_pred.shape == y_test.shape, f"Expected y_pred shape {y_test.shape}, got {y_pred.shape}"
    
    # Test intermediate representations extraction
    print("\nTesting intermediate representations extraction...")
    intermediate_reps = pipeline.extract_intermediate_representations(
        X_test,
        intermediate_model=selector.intermediate_representations['model']
    )
    
    # Check intermediate representations
    assert intermediate_reps.shape == (len(X_test), 32), \
        f"Expected intermediate_reps shape ({len(X_test)}, 32), got {intermediate_reps.shape}"
    
    # Test softmax outputs extraction
    print("\nTesting softmax outputs extraction...")
    softmax_outputs = pipeline.extract_softmax_outputs(
        X_test,
        softmax_model=selector.softmax_outputs['model']
    )
    
    # Check softmax outputs
    assert softmax_outputs.shape == (len(X_test), n_classes), \
        f"Expected softmax_outputs shape ({len(X_test)}, {n_classes}), got {softmax_outputs.shape}"
    
    print("\n✓ TrainingPipeline tests passed")
    return True, intermediate_reps, softmax_outputs, y_test, y_pred


def test_residuals_analyzer(intermediate_reps, softmax_outputs, y_test, y_pred):
    """Test the ResidualsAnalyzer class."""
    print("\n=== Testing ResidualsAnalyzer ===")
    
    # Create residuals analyzer
    analyzer = ResidualsAnalyzer(save_dir='./test_results/residuals')
    
    # Test confusion matrix generation
    print("\nTesting confusion matrix generation...")
    cm = analyzer.generate_confusion_matrix(y_test, y_pred)
    
    # Check confusion matrix
    n_classes = len(np.unique(np.concatenate([y_test, y_pred])))
    assert cm.shape == (n_classes, n_classes), f"Expected confusion matrix shape ({n_classes}, {n_classes}), got {cm.shape}"
    
    # Test cosine similarity calculation
    print("\nTesting cosine similarity calculation...")
    similarity_matrices = analyzer.calculate_cosine_similarity(intermediate_reps, y_pred)
    
    # Check similarity matrices
    assert isinstance(similarity_matrices, dict), f"Expected dict, got {type(similarity_matrices)}"
    
    # Test average similarity calculation
    print("\nTesting average similarity calculation...")
    average_similarities = analyzer.calculate_average_similarity(similarity_matrices)
    
    # Check average similarities
    assert isinstance(average_similarities, dict), f"Expected dict, got {type(average_similarities)}"
    assert set(average_similarities.keys()) == set(similarity_matrices.keys()), \
        "Keys in average_similarities and similarity_matrices should match"
    
    # Test entropy calculation
    print("\nTesting entropy calculation...")
    entropies = analyzer.calculate_entropy(softmax_outputs)
    
    # Check entropies
    assert entropies.shape == (len(softmax_outputs),), f"Expected entropies shape ({len(softmax_outputs)},), got {entropies.shape}"
    
    # Test correlation analysis
    print("\nTesting correlation analysis...")
    correlation_stats = analyzer.analyze_correlation(average_similarities, entropies, y_pred)
    
    # Check correlation stats
    assert isinstance(correlation_stats, dict), f"Expected dict, got {type(correlation_stats)}"
    
    print("\n✓ ResidualsAnalyzer tests passed")
    return True, similarity_matrices, average_similarities, entropies


def test_visualization_analyzer(intermediate_reps, softmax_outputs, y_test, y_pred, 
                              similarity_matrices, average_similarities, entropies):
    """Test the VisualizationAnalyzer class."""
    print("\n=== Testing VisualizationAnalyzer ===")
    
    # Create visualization analyzer
    visualizer = VisualizationAnalyzer(save_dir='./test_results/visualization')
    
    # Test correlation heatmap
    print("\nTesting correlation heatmap...")
    visualizer.plot_correlation_heatmap(
        similarity_matrices,
        entropies,
        y_pred,
        save_path='./test_results/visualization/correlation_heatmap.png'
    )
    
    # Check if file was created
    assert os.path.exists('./test_results/visualization/correlation_heatmap.png'), \
        "Correlation heatmap file not created"
    
    # Test class performance comparison
    print("\nTesting class performance comparison...")
    visualizer.plot_class_performance_comparison(
        y_test,
        y_pred,
        entropies,
        save_path='./test_results/visualization/class_performance.png'
    )
    
    # Check if file was created
    assert os.path.exists('./test_results/visualization/class_performance.png'), \
        "Class performance file not created"
    
    # Test similarity-entropy joint distribution
    print("\nTesting similarity-entropy joint distribution...")
    visualizer.plot_similarity_entropy_joint(
        average_similarities,
        entropies,
        y_pred,
        save_path='./test_results/visualization/similarity_entropy_joint.png'
    )
    
    # Check if file was created
    assert os.path.exists('./test_results/visualization/similarity_entropy_joint.png'), \
        "Similarity-entropy joint distribution file not created"
    
    # Test statistical significance
    print("\nTesting statistical significance...")
    visualizer.plot_statistical_significance(
        average_similarities,
        entropies,
        y_pred,
        save_path='./test_results/visualization/statistical_significance.png'
    )
    
    # Check if file was created
    assert os.path.exists('./test_results/visualization/statistical_significance.png'), \
        "Statistical significance file not created"
    
    print("\n✓ VisualizationAnalyzer tests passed")
    return True


def main():
    """Run all tests."""
    print("=== Running Tests for ML Simulation Environment ===")
    
    # Create test results directory
    os.makedirs('./test_results/residuals', exist_ok=True)
    os.makedirs('./test_results/visualization', exist_ok=True)
    
    # Test ImageDataGenerator
    test_image_data_generator()
    
    # Test ModelSelector
    test_model_selector()
    
    # Test TrainingPipeline
    training_success, intermediate_reps, softmax_outputs, y_test, y_pred = test_training_pipeline()
    
    # Test ResidualsAnalyzer
    residuals_success, similarity_matrices, average_similarities, entropies = test_residuals_analyzer(
        intermediate_reps, softmax_outputs, y_test, y_pred
    )
    
    # Test VisualizationAnalyzer
    visualization_success = test_visualization_analyzer(
        intermediate_reps, softmax_outputs, y_test, y_pred,
        similarity_matrices, average_similarities, entropies
    )
    
    print("\n=== All Tests Completed Successfully ===")
    print("The ML Simulation Environment is working correctly!")


if __name__ == "__main__":
    main()
