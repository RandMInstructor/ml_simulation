# ML Simulation Environment with Residuals Analysis

A comprehensive simulation environment for demonstrating machine learning solutions with residuals analysis for multi-class classification problems, with a focus on image classification.

## Features

- **Customizable Data Generation**: Generate synthetic data for multi-class classification with various methods
  - Standard data generation (blobs, classification, moons, circles, gaussian mixture)
  - Image data generation (shapes, patterns, noise, gradients)

- **Flexible Model Selection**: Choose from multiple machine learning architectures
  - Deep Learning: MLP, CNN, RNN
  - Traditional ML: Random Forest, Gradient Boosting, SVM, Logistic Regression
  - Customizable hyperparameters for each model type

- **Comprehensive Training Pipeline**: Train and evaluate models with detailed metrics
  - Training with validation
  - Model checkpointing and early stopping
  - Detailed evaluation metrics and confusion matrices

- **Advanced Residuals Analysis**: Analyze model performance beyond accuracy
  - Extract intermediate representations from deep learning models
  - Calculate pairwise cosine similarity between representations
  - Compute entropy from softmax outputs
  - Analyze correlation between similarity and entropy

- **Statistical Tests and Visualization**: Understand model behavior with statistical rigor
  - Both Pearson and Spearman rank correlation coefficients
  - Confidence intervals and significance testing
  - Comprehensive visualization of results

## Installation

```bash
# Clone the repository
git clone https://github.com/ml-simulation/ml-simulation.git
cd ml-simulation

# Install the package
pip install -e .
```

## Quick Start

Run the demonstration script with default parameters:

```bash
python demo.py
```

Customize the demonstration with command-line arguments:

```bash
python demo.py --data_method shapes --n_classes 5 --model_type cnn --hidden_layers 128,64 --epochs 30
```

## Usage Examples

### Generate Synthetic Image Data

```python
from ml_simulation import ImageDataGenerator

# Create data generator
generator = ImageDataGenerator(random_state=42)

# Generate images
X, y = generator.generate_image_data(
    method='shapes',  # 'shapes', 'patterns', 'noise', 'gradients'
    n_samples=1000,
    n_classes=5,
    image_size=(32, 32)
)

# Preprocess and split data
X_processed = generator.preprocess_images(X)
data_splits = generator.split_data(X_processed, y)

# Visualize sample images
generator.visualize_images(X, y, n_images_per_class=5)
```

### Create and Train a Model

```python
from ml_simulation import ModelSelector, TrainingPipeline
import tensorflow as tf

# Create model selector
selector = ModelSelector(random_state=42)

# Get model
model = selector.get_model(
    model_type='cnn',  # 'mlp', 'cnn', 'rnn', 'random_forest', etc.
    n_classes=5,
    n_features=32*32,  # Flattened input shape
    hidden_layers=[128, 64]
)

# Create training pipeline
pipeline = TrainingPipeline(model=model, model_type='keras')

# Train model
history = pipeline.train(
    X_train=data_splits['X_train'],
    y_train=data_splits['y_train'],
    X_val=data_splits['X_val'],
    y_val=data_splits['y_val'],
    batch_size=32,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5')
    ]
)

# Evaluate model
metrics = pipeline.evaluate(X_test=data_splits['X_test'], y_test=data_splits['y_test'])
```

### Perform Residuals Analysis

```python
from ml_simulation import ResidualsAnalyzer

# Get predictions
y_pred = pipeline.predict(data_splits['X_test'])

# Extract intermediate representations
intermediate_reps = pipeline.extract_intermediate_representations(
    data_splits['X_test'],
    intermediate_model=selector.intermediate_representations['model']
)

# Extract softmax outputs
softmax_outputs = pipeline.extract_softmax_outputs(
    data_splits['X_test'],
    softmax_model=selector.softmax_outputs['model']
)

# Create residuals analyzer
analyzer = ResidualsAnalyzer(save_dir='./results')

# Run full analysis
results = analyzer.run_full_analysis(
    representations=intermediate_reps,
    softmax_outputs=softmax_outputs,
    y_true=data_splits['y_test'],
    y_pred=y_pred
)
```

### Visualize Results with Statistical Tests

```python
from ml_simulation import VisualizationAnalyzer

# Create visualization analyzer
visualizer = VisualizationAnalyzer(save_dir='./visualization')

# Run full visualization with statistical tests
visualization_results = visualizer.run_full_visualization(
    representations=intermediate_reps,
    softmax_outputs=softmax_outputs,
    y_true=data_splits['y_test'],
    y_pred=y_pred
)
```

## Command-line Arguments

The demonstration script (`demo.py`) supports the following command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_method` | Method for generating synthetic image data | `shapes` |
| `--n_samples` | Number of samples to generate | `1000` |
| `--n_classes` | Number of classes | `5` |
| `--image_size` | Size of the generated images (square) | `32` |
| `--model_type` | Type of model to use | `cnn` |
| `--hidden_layers` | Comma-separated list of neurons in hidden layers | `128,64` |
| `--batch_size` | Batch size for training | `32` |
| `--epochs` | Number of epochs for training | `30` |
| `--learning_rate` | Learning rate for optimizer | `0.001` |
| `--output_dir` | Directory to save results | `./results` |
| `--random_seed` | Random seed for reproducibility | `42` |

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow 2.4+
- pandas
- seaborn
- SciPy

## License

MIT License
