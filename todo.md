# ML Simulation Environment with Residuals Analysis

## Tasks

- [x] Create project structure
- [x] Implement data generation module
  - [x] Create customizable methods for synthetic data generation
  - [x] Support multiple classes with different distributions
- [x] Implement model selection module
  - [x] Support different ML architectures
  - [x] Allow hyperparameter customization
- [x] Implement training and evaluation pipeline
  - [x] Create training loop
  - [x] Implement validation process
  - [x] Save model states
- [x] Implement residuals analysis module
  - [x] Generate confusion matrix
  - [x] Extract intermediate representations
  - [x] Calculate pairwise cosine similarity
  - [x] Calculate entropy from SoftMax
- [x] Implement visualization and statistical tests
  - [x] Plot correlation between cosine similarity and entropy
  - [x] Perform statistical tests for correlation
  - [x] Visualize results for each class
- [x] Create demonstration script
  - [x] Showcase all features
  - [x] Document usage examples
- [] Test and finalize simulation environment
  - [] Verify all requirements are met
  - [] Ensure code quality and documentation
- [] Perform comprehensive output assessment and error verification
  - [] Investigate statistical test results showing only class 3 entries
  - [] Document final verification results and potential improvements




We need to continue work on the simulation environment project that is in the attached zip file.The project is implemented in python to demonstrate a machine learning solution with residuals analysis for a multi-class classification problem. The simulation environment generates synthetic image data based upon the requested number of classes with customizable methods to generate examples for each class in the demonstration data set. The environment supports the selection of a machine learning architecture and associated hyperparameters for training and testing. For basic residuals analysis, a confusion matrix is generated to visualize the performance of the selected model on each of the classes represented in the data set. For advanced residuals analysis, models with intermediate representations before the final prediction layer, or output of class labels for each observation, the layer state should be recorded in parallel with the final predicted class label. These stored representations should then be used to compute pairwise cosine similarity among the stored representations for each observation predicted to be in the same class by the final trained model under evaluation. The entropy calculated from the SoftMax function for the model is also recorded for each observation and prediction. Finally, Spearman's rank correlation coefficient is used as a statistical test for analyzing the relationship between average pairwise cosine similarity and entropy values in the advanced residuals analysis portion of the demonstration. The average pairwise cosine similarity between each observation and all like-class assigned observations and the entropy value for each prediction are also plotted for each class in the generated data set along with the statistical test results including a trend line for the correlation with a shaded confidence interval for correlation between the distance and entropy values.

The code and initial results need to be verified for one of the demo cases.
enhanced_demo.py --data_method shapes --n_classes 5 --model_type cnn --hidden_layers 128,64 --epochs 50 --n_samples 100 --enhanced_visualization
Perform comprehensive output assessment and error verification
Investigate statistical test results showing only class 3 entries
Document final verification results and potential improvements
