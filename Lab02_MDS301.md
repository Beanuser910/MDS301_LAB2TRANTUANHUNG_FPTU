# MACHINE LEARNING IN DATA SCIENCE

## LAB 02

**Building and Training Neural Networks with Scikit-Learn & PyTorch**

**Tools:** Python 3.x, Scikit-learn, PyTorch, TorchVision

| Student Name | Student RollNumber |
|---|---|
| _______________________________ | _______________________________ |

# Lab Overview & Objectives

This practical lab consolidates the core skills concerning neural networks of the Machine Learning in Data Sciences course. Students will implement, train, evaluate, and optimise neural network models using Scikit-Learn and PyTorch, culminating in a transfer-learning exercise on a real-world image classification task. The learning objectives include:

- Implement and train Multilayer Perceptron (MLP) models for regression and classification tasks using Scikit-Learn.
- Build and train Convolutional Neural Networks (CNN) using PyTorch.
- Implement residual connections and understand the ResNet architecture.

# Grading Rubric

| Section | Points | Criteria |
|---|---:|---|
| Part 1: Regression MLP | 30 pts | Correct implementation & insight of analysis |
| Part 2: Classification MLP | 30 pts | Correct implementation & insight of analysis |
| Part 3: Deep Learning with PyTorch | 40 pts | Correct implementation & insight of analysis |
| TOTAL | 100 pts |  |

## Prerequisites & Setup

Before starting the lab, ensure your environment is configured correctly. Run the following cell in your Jupyter Notebook.

```python
# Run this cell to verify your environment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

print('PyTorch version:', torch.__version__)
print('CUDA available: ', torch.cuda.is_available())
```

Note that if you don’t have a GPU on your computer, Google Colab which provides free GPU for accelerating the training is strongly recommended for part 3 (change runtime type to GPU before running any code).

## Submission guidelines:

Submit a single Jupyter Notebook (.ipynb) file containing all code, outputs, and written answers. Include your full name and student ID in the first cell.

# PART 1 — Regression MLP (30 pts)

## 1.1  Data Preparation  [5 pts]

- Load the California Housing dataset using fetch_california_housing().
- Split into train/test sets (80%/20%) with random_state=42.
- Apply StandardScaler — fit on training data, transform both sets.
- Print the shape of each split and report the feature names

## 1.2  Baseline MLP Regressor  [10 pts]

- Instantiate an MLPRegressor with one hidden layer of 64 neurons, ReLU activation, Adam optimiser, and max_iter=500.
- Train the model and evaluate on the test set using RMSE and R² score.
- Plot the learning curve (loss vs. iteration) using loss_curve_.
- Discuss: what do the curve's shape and plateau suggest about convergence?

## 1.3  Hyperparameter Tuning  [15 pts]

- Run a GridSearchCV over the following grid (5-fold CV, scoring=neg_root_mean_squared_error). Pass n_jobs=-1 to GridSearchCV to parallelise across all CPU cores (≈5–8 min on CPU; ≈30 min without parallelisation).

```python
param_grid = {
    'hidden_layer_sizes': [(64,), (128, 64), (256, 128, 64)],
    'activation':         ['relu', 'tanh'],
    'learning_rate_init': [0.01, 0.001],
    'alpha':              [0.0001, 0.001],   # L2 regularisation
}
```

- Report the best parameters and corresponding CV score.
- Evaluate the best estimator on the test set; compare it with the baseline.
- In 3–5 sentences, explain the effect of depth (number of hidden layers) and L2 regularisation (alpha) on the results.

# PART 2 — Classification MLP  (30 pts)

Use the Breast Cancer Wisconsin dataset (binary classification) loaded via `load_breast_cancer()`.

## 2.1  Data Exploration & Preparation  [5 pts]

- Print class distribution and check for missing values.
- Apply StandardScaler and perform the train/test split.
- Explain why feature scaling is important for gradient-based optimisers.

## 2.2  MLP Classifier  [10 pts]

- Instantiate an MLPClassifier and choose an appropriate architecture (number of hidden layers, neurons per layer, activation function). Justify your design choices in 2–3 sentences.
- Train the model and evaluate on the test set using accuracy score, precision, recall, and F1-score. Report the full classification_report.
- Plot the learning curve and the confusion matrix. In 3–5 sentences, analyse the results: discuss precision vs. recall trade-off and whether the model generalises well.

## 2.3  Optimiser Comparison  [8 pts]

- Train three identical architectures using SGD, Adam, and L-BFGS solvers.
- Plot all three loss curves on the same figure.
- Discuss the trade-offs between the solvers in terms of speed, stability, and final loss.

## 2.4  Early Stopping  [7 pts]

In this section you will implement early stopping in MLPClassifier and empirically compare learning curves before and after. (~5 minutes on CPU)

- Train two MLPClassifiers with the same architecture from Section 2.2:
   - (a) baseline with early_stopping=False, max_iter=500;
   - (b) with early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=500.

- Plot both training loss curves (loss_curve_) on the same figure. Mark the epoch at which early stopping triggered for model (b) with a vertical dashed line.
- Report test accuracy and total iterations used by each model. In 3–5 sentences, discuss: Did early stopping improve generalisation? How many iterations were saved? What are the trade-offs?

# PART 3 — Deep Learning with PyTorch  (40 pts)

This part focuses on building CNNs from scratch and understanding modern architectural components using PyTorch.

## 3.1  PyTorch Fundamentals Review  [5 pts]

Complete the following short exercises as warm-up:

- Create a random tensor of shape (3, 224, 224) representing an RGB image. Normalise it to zero mean and unit variance. Print shape, dtype, mean, and standard deviation.
- Demonstrate hardware acceleration: create two large tensors (2048 × 2048) and time a matrix multiplication on CPU vs. GPU (if available). Report the speed-up factor.
- Implement a simple computation graph for the function f(x) = (x² + 3x – 2)³, compute df/dx at x = 2 using autograd, and verify analytically.

## 3.2  CNN for CIFAR-10 Image Classification  [20 pts]

### 3.2.1  Data Loading & Augmentation  [4 pts]

- Download CIFAR-10 using torchvision.datasets.CIFAR10.
- Apply training augmentations: random horizontal flip, random crop (32×32 with padding 4), and normalise with CIFAR-10 statistics.
- Create DataLoader objects (batch size 64, num_workers=2). Display a grid of 16 sample images with their class labels.

### 3.2.2  Custom CNN Architecture  [8 pts]

Implement the CNN architecture described below as a PyTorch `nn.Module`:

```python
class CustomCNN(nn.Module):
    """
    Block 1 : Conv(3→32, 3×3, pad=1) → BN → ReLU → Conv(32→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
    Block 2 : Conv(32→64, 3×3, pad=1) → BN → ReLU → Conv(64→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
    Block 3 : Conv(64→128, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool(1×1)
    Classifier: Flatten → Dropout(0.5) → Linear(128→256) → ReLU → Linear(256→10)
    """
    pass   # your implementation here
```

- Print the model summary (total trainable parameters).
- Verify a forward pass with a dummy batch of shape (4, 3, 32, 32).

### 3.2.3  Training Loop  [8 pts]

- Train for 30 epochs using Adam (lr=0.001) and Cross-Entropy Loss.
- Implement a step learning rate scheduler that decays by 0.1 every 10 epochs.
- Record and plot training/validation loss and accuracy curves.
- Report final test accuracy. Discuss: does the model overfit? How can regularisation help?

## 3.3  Residual Connections  [15 pts]

### 3.3.1  Residual Block Implementation  [7 pts]

Implement a standard ResNet residual block and a bottleneck residual block:

```python
class ResidualBlock(nn.Module):
    """
    Conv(C→C, 3×3, pad=1) → BN → ReLU
    Conv(C→C, 3×3, pad=1) → BN
    + skip connection → ReLU
    Handle dimension mismatch with projection shortcut (1×1 conv).
    """
    pass

class BottleneckBlock(nn.Module):
    """
    Conv(C→C//4, 1×1) → BN → ReLU
    Conv(C//4→C//4, 3×3, pad=1) → BN → ReLU
    Conv(C//4→C, 1×1) → BN
    + skip connection → ReLU
    """
    pass
```

Verify that gradients flow through the skip connections by printing gradient norms at each layer for a synthetic backward pass.

### 3.3.2  Mini ResNet on CIFAR-10  [8 pts]

- Build a 20-layer ResNet using your ResidualBlock (following the ResNet-20 layout: 3 groups of 3 blocks with 16, 32, 64 filters).
- Train for 30 epochs with SGD (lr=0.1, momentum=0.9, weight_decay=1e-4) and cosine annealing LR schedule.
- Compare test accuracy, parameter count, and training time with Task B2's CustomCNN.
- Visualise the feature maps of the first convolutional layer for three test images.

### Background: Learning Rate Schedulers

A learning rate scheduler adjusts the learning rate during training. A fixed learning rate is often suboptimal: too high a rate causes instability late in training, while too low a rate slows convergence early on. PyTorch provides schedulers in torch.optim.lr_scheduler. Three common strategies are:

(1) StepLR — decays the LR by a fixed factor every N epochs;  
(2) CosineAnnealingLR — smoothly decreases the LR following a cosine curve, reaching a near-zero minimum at T_max epochs;  
(3) OneCycleLR — warms up the LR then decays it, useful for fast convergence.  

Example using cosine annealing:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
for epoch in range(30):                 # inside your training loop
    train_one_epoch(model, optimizer, ...)  # your training step
    scheduler.step()                        # update LR after each epoch
```
