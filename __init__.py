"""
Main package initialization file for ML Simulation Environment.

This file initializes the ML Simulation Environment package and
provides convenient imports for the main components.
"""

from src.data_generator import DataGenerator
from src.image_data_generator import ImageDataGenerator
from src.model_selector import ModelSelector
from src.training_pipeline import TrainingPipeline
from src.residuals_analyzer import ResidualsAnalyzer
from src.visualization_analyzer import VisualizationAnalyzer

__version__ = '1.0.0'
__author__ = 'ML Simulation Team'
