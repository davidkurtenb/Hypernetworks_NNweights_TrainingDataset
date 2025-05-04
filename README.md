# Hypernetworks Neural Network Weights Training Dataset

## Summary
This repository contains a dataset of neural networks, designed for the purpose of hypernetworks research. The dataset includes 10,610 neural networks trained for binary image classification separated into 10 classes, such that each class contains 1,061 different neural networks that can identify a certain ImageNette V2 class from all other classes. The classification models used a LeNet-5 framework with each model containing 91,481 parameters. A computing cluster of over 10,000 cores was used to generate the dataset. Basic classification results show that the neural networks can be classified with accuracy of 87%, indicating that the differences between the neural networks can be identified by supervised machine learning algorithms in accuracy better than mere chance. The ultimate purpose of the dataset is to enable hypernetworks research.

## Specifications
This repository contains the individual model files, consolidated listing of parameters, and training/validation loss and accuracy plots for each model.

### Directory Listing
* **Model Weights** - contains individual model parameters saved as hdf5 files
* **Plots** - contains plots for training/validation loss and accuracy for each model 
* **Pickle Files** - consolidated parameters saved into a single file for all 10,610 models. 
   * *Modelwise* - parameters by model in dictionary format
      * *Keys*: model, parameters_by_layer, parameters_flat 
      * *Example*: {'model': 'church_649', 'parameters_by_layer': {'conv2d/conv2d/bias:0': tensor([ 0.0572, 0.0594, -0.0007, -0.0440, 0.0227, 0.0155]), 'conv2d/conv2d/kernel:0': tensor([[[[ 2.0242e-01, -8.1340e-02, -1.1604e-01, 6.9871e-02, -1.4720e-01,...}, 'parameters_flat': tensor([ 0.0572, 0.0594, -0.0007, ..., 0.2816, 0.3210, -0.1104])}
   * *Layerwise* - parameters by layer for all models in a given class
      * *Keys*: class, parameters 
      * *Example*: {'class': 'cassette_player', 'parameters': {'conv2d': [array([ 2.81625651e-02, -7.09968358e-02, -1.40507936e-01, -9.32385959e-03, -1.10361770e-01, 1.00062557e-01, -3.23645794e-03, 1.01349339e-01, 5.19516990e-02, -8.61838460e-02, 1.42671123e-01, -1.31018981e-01,...'dense_1': [array([-0.04802695, -0.26561895, -0.13131969,  0.13551578, -0.14921758, -0.15271764,  0.226575...}

## Performance Metrics by Class

| Class | Accuracy Min | Accuracy Max | Accuracy Avg | Precision Min | Precision Max | Precision Avg | Recall Min | Recall Max | Recall Avg | F1 Min | F1 Max | F1 Avg |
|-------|--------------|--------------|--------------|---------------|---------------|---------------|------------|------------|------------|--------|--------|--------|
| **tench** | 0.932 | 0.949 | 0.942 | 0.663 | 0.872 | 0.769 | 0.478 | 0.672 | 0.587 | 0.604 | 0.713 | 0.665 |
| **english_springer** | 0.894 | 0.920 | 0.911 | 0.473 | 0.779 | 0.619 | 0.134 | 0.496 | 0.315 | 0.219 | 0.522 | 0.412 |
| **cassette_player** | 0.916 | 0.937 | 0.928 | 0.544 | 0.845 | 0.675 | 0.272 | 0.569 | 0.408 | 0.407 | 0.581 | 0.506 |
| **chain_saw** | 0.897 | 0.908 | 0.903 | 0.349 | 0.933 | 0.576 | 0.008 | 0.127 | 0.068 | 0.015 | 0.214 | 0.120 |
| **church** | 0.901 | 0.921 | 0.911 | 0.533 | 0.844 | 0.666 | 0.134 | 0.438 | 0.301 | 0.226 | 0.518 | 0.411 |
| **french_horn** | 0.886 | 0.907 | 0.900 | 0.300 | 0.634 | 0.507 | 0.008 | 0.353 | 0.186 | 0.015 | 0.406 | 0.265 |
| **garbage_truck** | 0.892 | 0.927 | 0.917 | 0.464 | 0.846 | 0.645 | 0.193 | 0.584 | 0.395 | 0.303 | 0.565 | 0.484 |
| **gas_pump** | 0.870 | 0.901 | 0.892 | 0.295 | 0.684 | 0.480 | 0.062 | 0.234 | 0.151 | 0.109 | 0.306 | 0.228 |
| **golf_ball** | 0.898 | 0.919 | 0.912 | 0.496 | 0.836 | 0.658 | 0.128 | 0.434 | 0.298 | 0.222 | 0.494 | 0.407 |
| **parachute** | 0.920 | 0.944 | 0.937 | 0.583 | 0.875 | 0.773 | 0.313 | 0.674 | 0.532 | 0.448 | 0.685 | 0.626 |
| **All Classes** | **0.870** | **0.949** | **0.915** | **0.295** | **0.933** | **0.637** | **0.008** | **0.674** | **0.324** | **0.015** | **0.713** | **0.412** |
