# CIFAR-100 CNN Classifier

A convolutional neural network built with PyTorch for image classification on the CIFAR-100 dataset.

## Overview

This project implements a CNN image classifier from scratch using PyTorch. The model uses reusable convolutional blocks consisting of convolution, batch normalization, ReLU activation, and max pooling.

The project covers the complete deep learning workflow:

* Dataset loading and preprocessing
* Data augmentation
* CNN architecture design
* Model training
* Validation
* Performance evaluation
* Prediction visualization
* Per-class accuracy analysis
* Model saving

## Dataset

The project uses the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

* 60,000 RGB images
* Image size: 32 × 32
* 100 classes
* 50,000 training images
* 10,000 test images

## Model Architecture

The CNN consists of three convolutional blocks followed by a fully connected classification head.

```text
Input
3 × 32 × 32
     │
     ▼
CNN Block 1
3 → 32 channels
32 × 32 → 16 × 16
     │
     ▼
CNN Block 2
32 → 64 channels
16 × 16 → 8 × 8
     │
     ▼
CNN Block 3
64 → 128 channels
8 × 8 → 4 × 4
     │
     ▼
Flatten
128 × 4 × 4 = 2048
     │
     ▼
Linear
2048 → 512
     │
     ▼
ReLU
     │
     ▼
Dropout
     │
     ▼
Linear
512 → 100 classes
```

### CNN Block

Each convolutional block contains:

```text
Conv2d
BatchNorm2d
ReLU
MaxPool2d
```

## Data Augmentation

Training images are augmented using:

* Random horizontal flip
* Random vertical flip
* Random rotation

CIFAR-100 normalization is also applied before training.

## Training Configuration

| Parameter     | Value             |
| ------------- | ----------------- |
| Framework     | PyTorch           |
| Dataset       | CIFAR-100         |
| Batch Size    | 64                |
| Optimizer     | Adam              |
| Learning Rate | 0.0005            |
| Weight Decay  | 0.0005            |
| Loss Function | CrossEntropyLoss  |
| Epochs        | 50                |
| Device        | CUDA if available |

## Results

Results will be added after training.

| Metric                   | Result |
| ------------------------ | -----: |
| Best Validation Accuracy |    TBD |
| Test Accuracy            |    TBD |

## Training Curves

### Loss

Results visualization will be added here.

### Accuracy

Results visualization will be added here.

## Predictions

Example model predictions will be added here.

## Technologies

* Python
* PyTorch
* Torchvision
* Matplotlib

## License

This project is licensed under the MIT License.
