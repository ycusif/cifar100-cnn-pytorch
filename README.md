# CIFAR-100 Image Classification with PyTorch

A comparative deep learning study of image classification on the **CIFAR-100** dataset using custom convolutional neural networks and ResNet-18 transfer learning.

The project focuses on model architecture, data-splitting methodology, augmentation, transfer learning, validation practices, and per-class performance analysis.

## Overview

CIFAR-100 contains **60,000 32×32 RGB images** across **100 classes**, making it a useful benchmark for evaluating image classification models.

This project was developed through three progressive experiments:

| Version | Model                       | Best Validation Accuracy | Test Accuracy |
| ------- | --------------------------- | -----------------------: | ------------: |
| V1      | Basic CNN                   |                  50.15%* |       50.15%* |
| V2      | Improved CNN                |                   64.64% |        64.20% |
| V3      | ResNet-18 Transfer Learning |               **80.40%** |    **78.97%** |

* V1 used the official CIFAR-100 test set as validation, so this result is not treated as a clean validation benchmark. V2 corrected this evaluation methodology by creating a dedicated validation split.

The final experiment, **V3**, uses an ImageNet-pretrained ResNet-18 and achieves **78.97% test accuracy**.

---

## Dataset

The project uses the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Dataset structure:

* 50,000 training images
* 10,000 test images
* 100 classes
* 600 images per class
* 32×32 RGB images

### Data Split

For V2 and V3, the official training set was divided into:

* **45,000 training images**
* **5,000 validation images**
* **10,000 test images**

The validation split was created using a fixed random seed:

```python
torch.Generator().manual_seed(42)
```

The official test set remained untouched until final evaluation.

This prevents the test set from influencing model selection.

---

# Experiments

## V1: Baseline CNN

The first experiment used a relatively simple CNN:

```text
Input: 3 × 32 × 32

Conv Block
3 → 32

Conv Block
32 → 64

Conv Block
64 → 128

Flatten
128 × 4 × 4 = 2048

Fully Connected
2048 → 512 → 100
```

Each convolutional block contained:

* 3×3 convolution
* Batch normalization
* ReLU activation
* 2×2 max pooling

### V1 Results

Approximately **50.15% accuracy** was obtained.

However, the evaluation methodology was later identified as a weakness because the official CIFAR-100 test set was being used as the validation set.

This motivated the second experiment.

---

# V2: Improved CNN

The second experiment improved both the architecture and evaluation methodology.

### Architecture

```text
Input: 3 × 32 × 32

Block 1
3 → 64

Block 2
64 → 128

Block 3
128 → 256

Flatten
256 × 4 × 4 = 4096

Fully Connected
4096 → 512 → 100
```

Each convolutional block used two convolutional layers followed by batch normalization, ReLU activations, and max pooling.

### Data Augmentation

Training images were augmented using:

* Random horizontal flip
* Random rotation
* CIFAR-100 normalization

Vertical flipping was removed because it is generally less appropriate for natural images.

### Results

* Best validation accuracy: **64.64%**
* Test accuracy: **64.20%**

The improvement over V1 demonstrated the impact of both architectural changes and cleaner evaluation methodology.

---

# V3: ResNet-18 Transfer Learning

The final experiment uses **ResNet-18 pretrained on ImageNet**.

Instead of training the network entirely from scratch, the pretrained model was adapted to the 100-class CIFAR-100 classification problem.

### Model

```text
ImageNet-pretrained ResNet-18
            ↓
      Replace FC layer
            ↓
       512 → 100
```

The original ImageNet classification layer was replaced with:

```python
nn.Linear(512, 100)
```

All ResNet-18 parameters were fine-tuned during training.

### Input Processing

Because standard ResNet-18 expects larger ImageNet-style inputs, CIFAR-100 images were resized:

```text
32 × 32 → 224 × 224
```

Training augmentation:

* Random horizontal flip
* Random rotation up to 15°
* ImageNet normalization

Validation and test images were resized and normalized without random augmentation.

### Training Configuration

| Parameter          |               Value |
| ------------------ | ------------------: |
| Model              |           ResNet-18 |
| Initialization     | ImageNet pretrained |
| Optimizer          |                Adam |
| Learning rate      |              0.0001 |
| Weight decay       |              0.0001 |
| Batch size         |                  64 |
| Epochs             |                  20 |
| Loss               |  Cross-Entropy Loss |
| Training samples   |              45,000 |
| Validation samples |               5,000 |
| Test samples       |              10,000 |

---

# V3 Results

The best validation performance occurred at **epoch 5**.

| Metric                   |                     Result |
| ------------------------ | -------------------------: |
| Best Validation Accuracy |                 **80.40%** |
| Final Test Accuracy      |                 **78.97%** |
| Validation-Test Gap      | **1.43 percentage points** |

The model was selected using validation accuracy, and the best checkpoint was saved before evaluating the untouched test set.

### Training Behavior

The model reached:

* **89.26% training accuracy** by epoch 5
* **80.40% validation accuracy** by epoch 5

By epoch 20:

* Training accuracy: **97.88%**
* Validation accuracy: **79.68%**

This indicates increasing overfitting after the early training stages.

The validation accuracy plateaued around 80%, while training accuracy continued approaching 98%.

---

# Comparison

The progression across experiments shows a substantial improvement:

```text
V1  Basic CNN
    ~50.15%
       ↓
V2  Improved CNN
    64.20%
       ↓
V3  ResNet-18 Transfer Learning
    78.97%
```

V3 improved test accuracy by:

**14.77 percentage points over V2.**

This demonstrates the effectiveness of using a deeper pretrained architecture compared with the custom CNN models used in the earlier experiments.

---

# Per-Class Analysis

Overall accuracy does not tell the complete story on a 100-class dataset.

The final ResNet-18 model was therefore evaluated on each CIFAR-100 class individually.

### Strongest Classes

Some of the highest-performing classes were:

| Class      | Accuracy |
| ---------- | -------: |
| Skunk      |      97% |
| Wardrobe   |      96% |
| Apple      |      95% |
| Orange     |      94% |
| Motorcycle |      94% |
| Road       |      94% |
| Tractor    |      94% |
| Bottle     |      93% |
| Skyscraper |      93% |
| Spider     |      93% |
| Sunflower  |      93% |

### Challenging Classes

Several classes were significantly harder:

| Class       | Accuracy |
| ----------- | -------: |
| Boy         |      31% |
| Man         |      51% |
| Oak Tree    |      54% |
| Girl        |      59% |
| Willow Tree |      59% |
| Otter       |      60% |
| Possum      |      62% |
| Beaver      |      63% |
| Maple Tree  |      63% |

The results suggest that classes involving visually similar objects, animals, and human categories remain more difficult for the model.

Per-class evaluation provides additional insight that cannot be obtained from overall accuracy alone.

---

# What I Learned

This project was developed incrementally rather than jumping directly to a pretrained model.

### 1. Evaluation methodology matters

The first experiment exposed an important problem: using the test set as validation can lead to misleading evaluation.

Creating a dedicated validation set produced a cleaner experimental setup.

### 2. Architecture has a major impact

Increasing the capacity and depth of the custom CNN improved performance from approximately 50% to 64%.

### 3. Transfer learning can provide a large performance advantage

Using ImageNet-pretrained ResNet-18 increased test accuracy from:

**64.20% → 78.97%**

This was the largest improvement across the experiments.

### 4. Overfitting must be monitored

Training accuracy continued increasing while validation accuracy plateaued.

This demonstrates why monitoring validation performance is important when selecting the final model.

### 5. Overall accuracy is not enough

Per-class evaluation revealed substantial differences between classes.

A model achieving nearly 79% overall accuracy can still perform poorly on individual categories.

---

# Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* CIFAR-100
* ResNet-18
* Transfer Learning
* Convolutional Neural Networks
* Data Augmentation
* Batch Normalization
* Model Checkpointing

---

# Project Structure

```text
cifar100-cnn-pytorch/
│
├── data/
│
├── train.py
│
├── README.md
│
├── .gitignore
│
└── LICENSE
```

`train.py` contains the final ResNet-18 training and evaluation pipeline.

---

# Future Improvements

Possible extensions to the experiment include:

* Compare frozen-backbone feature extraction against full fine-tuning
* Add learning-rate scheduling
* Introduce early stopping
* Experiment with stronger CIFAR-100 augmentation
* Evaluate precision, recall, and F1-score
* Generate a confusion matrix
* Investigate the hardest classes individually
* Compare ResNet-18 against larger architectures
* Experiment with CIFAR-specific ResNet architectures

---

# Conclusion

This project demonstrates a complete image-classification workflow using PyTorch, progressing from a baseline CNN to an improved custom architecture and finally to transfer learning with ResNet-18.

The final model achieved **78.97% accuracy on the CIFAR-100 test set**, while the experimental progression demonstrated how architecture selection, transfer learning, evaluation methodology, and overfitting analysis affect model performance.

The project emphasizes not only achieving a higher score, but understanding **why the model improved and where it still fails**.
