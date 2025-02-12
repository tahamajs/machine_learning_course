
# Machine Learning Course - Assignments & Final Project

This repository contains a series of assignments and a final project for the **Machine Learning Course** at the **University of Tehran**. The assignments cover both fundamental and advanced machine learning concepts including linear regression, logistic regression, classification, optimization, deep learning, and clustering.

## Table of Contents

1. [HW1: Cross-Validation &amp; Regularization](#1-hw1-cross-validation--regularization)
2. [HW2: Naïve Bayes &amp; Bayesian Decision Theory](#2-hw2-naïve-bayes--bayesian-decision-theory)
3. [HW3: Optimization Algorithms](#3-hw3-optimization-algorithms)
4. [HW4: Decision Trees &amp; Clustering](#4-hw4-decision-trees--clustering)
5. [HW5: Deep Learning &amp; Neural Networks](#5-hw5-deep-learning--neural-networks)
6. [Final Project: Speaker Gender Classification &amp; Voice Authentication](#6-final-project-speaker-gender-classification--voice-authentication)

---

## 1. HW1: Cross-Validation & Regularization

### Cross-Validation

- **Purpose:**Evaluate the performance of a machine learning model by partitioning the data into subsets. This helps in estimating how well the model generalizes to unseen data.
- **K-Fold Cross-Validation:**
  The dataset is divided into *k* equal-sized folds. For each iteration, one fold is used as the validation set and the remaining *k-1* folds are used for training. Averaging the performance over all *k* trials helps reduce variance and avoid overfitting.

### Regularization

- **Purpose:**Add a penalty to the loss function to discourage overly complex models, thereby preventing overfitting.
- **L1 Regularization (Lasso):**

  - **Mechanism:** Adds a penalty proportional to the absolute value of the coefficients.
  - **Effect:** Encourages sparsity by driving some coefficients to zero, resulting in simpler, more interpretable models.
- **L2 Regularization (Ridge):**

  - **Mechanism:** Adds a penalty proportional to the square of the coefficients.
  - **Effect:** Shrinks coefficients towards zero but typically does not force them to exactly zero, balancing the contribution of all features.

### Linear Regression Techniques

- **Closed-Form Solution (Normal Equation):**Directly computes the model parameters using matrix operations. Best suited for smaller datasets.
- **Gradient Descent:**
  An iterative optimization algorithm that updates parameters in the direction of the steepest decrease of the loss function. Particularly useful for large datasets.

---

## 2. HW2: Naïve Bayes & Bayesian Decision Theory

### Naïve Bayes Classifier

- **Fundamental Idea:**A probabilistic classifier based on Bayes’ theorem that assumes all features are conditionally independent given the class label. Despite its simplicity, it performs well in many applications.
- **Application:**
  Classifies data into discrete categories by computing the posterior probability for each class and selecting the one with the highest probability.

### Bayesian Decision Theory

- **Concept:**Provides a framework for making decisions under uncertainty by combining prior probabilities with evidence from the data.
- **Discriminant Functions:**Used to define decision boundaries between classes. In classification tasks, the class with the highest discriminant function value is chosen to minimize expected risk.
- **Gaussian-Distributed Classes:**
  When classes follow a Gaussian distribution, discriminant functions are derived based on the means and covariances of each class, helping visualize decision boundaries in the feature space.

---

## 3. HW3: Optimization Algorithms

### Gradient Descent

- **Concept:**An optimization algorithm that iteratively adjusts model parameters by moving opposite to the gradient of the loss function.
- **Step Size (Learning Rate):**
  Determines the size of the update steps. A large step size may overshoot the minimum, while a small step size may slow down convergence.

### Line Search Optimization

- **Purpose:**
  Finds an optimal step size along the gradient direction to enhance convergence speed and stability in gradient-based optimization.

### Newton’s Method

- **Overview:**A second-order optimization technique that uses both the gradient and the Hessian matrix (second derivatives) to find the function's minimum.
- **Advantages:**Typically converges faster than gradient descent when the Hessian is available and well-conditioned.
- **Limitations:**
  Computationally expensive for high-dimensional problems due to the Hessian computation.

### Quasi-Newton Methods (BFGS & DFP)

- **Purpose:**Approximate the Hessian (or its inverse) to achieve faster convergence without the high computational cost.
- **BFGS (Broyden–Fletcher–Goldfarb–Shanno) Algorithm:**Iteratively updates an approximation of the inverse Hessian using gradient information.
- **DFP (Davidon-Fletcher-Powell) Algorithm:**
  Similar to BFGS with a different update mechanism for the Hessian approximation.

---

## 4. HW4: Decision Trees & Clustering

### Decision Trees

- **Structure:**A flowchart-like model where each internal node tests a feature, each branch represents an outcome, and each leaf node represents a class label.
- **Splitting Criteria (Information Gain):**
  Measures the reduction in entropy after a split. The feature with the highest information gain is used to split the data, leading to purer child nodes.

### Bias-Variance Tradeoff

- **Bias:**Error due to overly simplistic assumptions, leading to underfitting.
- **Variance:**Error due to sensitivity to fluctuations in the training set, potentially causing overfitting.
- **Tradeoff:**
  Balancing bias and variance is key to minimizing overall prediction error.

### Clustering Algorithms

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**Groups points based on density, effectively identifying clusters of arbitrary shape and detecting outliers.
- **OPTICS (Ordering Points To Identify the Clustering Structure):**
  Similar to DBSCAN but capable of identifying clusters of varying densities without strict density parameter requirements.

---

## 5. HW5: Deep Learning & Neural Networks

### Multi-Layer Perceptron (MLP)

- **Structure:**A feedforward neural network with one or more hidden layers. Each neuron applies a nonlinear activation function.
- **Training (Backpropagation):**
  Weights are adjusted through backpropagation by computing the gradient of the loss function with respect to each weight.

### Convolutional Neural Networks (CNNs)

- **Architecture:**Specialized neural networks for grid-like data (e.g., images) that learn spatial hierarchies of features.
- **Key Components:**

  - **Convolutional Layers:** Apply filters to extract features from the input.
  - **Pooling Layers:** Reduce the spatial dimensions of feature maps, making the detection of features invariant to scale and translation.
  - **Fully Connected Layers:** Perform final classification based on the extracted features.

### XOR Classification & the Perceptron

- **XOR Problem:**A classic problem that is not linearly separable and cannot be solved by a single-layer perceptron.
- **Solution:**
  A multi-layer perceptron (MLP) with a hidden layer can solve the XOR problem by learning non-linear decision boundaries.

### Loss Functions & Optimization in Neural Networks

- **Loss Functions:**Measure the difference between predicted and actual values. Common examples include mean squared error (for regression) and cross-entropy (for classification).
- **Optimization:**
  Algorithms such as stochastic gradient descent (SGD), Adam, and RMSprop are used to minimize the loss by iteratively updating model parameters.

---

## 6. Final Project: Speaker Gender Classification & Voice Authentication

### Data Processing

- **Feature Extraction:**

  - **MFCC (Mel Frequency Cepstral Coefficients):**Capture the short-term power spectrum of a sound and are key features for speech recognition.
  - **Spectral Centroid:**Indicates the “center of mass” of the spectrum, which relates to the brightness of the sound.
  - **Zero-Crossing Rate:**
    Measures how frequently the signal changes sign, providing insights into the signal's frequency content.
- **Preprocessing Steps:**

  - **Data Normalization:**Scale features to ensure they contribute equally during model training.
  - **Noise Removal:**
    Apply filtering techniques to eliminate background noise that may distort feature extraction.
- **Feature Selection & Transformation:**
  Selecting the most relevant features and applying transformations (e.g., PCA) can improve model performance.

### Modeling

- **Classification Models:**

  - **SVM (Support Vector Machine):**Finds the optimal hyperplane that maximizes the margin between classes.
  - **MLP (Multi-Layer Perceptron):**Learns complex patterns through hidden layers.
  - **XGBoost:**
    A powerful gradient boosting algorithm that builds an ensemble of decision trees for high predictive accuracy.
- **Clustering:**
  Unsupervised techniques (e.g., DBSCAN or OPTICS) can be used to group similar speakers, aiding in voice authentication.

### Evaluation Metrics

- **ROC Curves:**Plot the true positive rate against the false positive rate at various threshold settings.
- **Precision, Recall, and F1-Score:**Evaluate model accuracy, especially in imbalanced datasets:
  - **Precision:** Fraction of relevant instances among those predicted.
  - **Recall:** Fraction of relevant instances that were correctly predicted.
  - **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.

---

## Installation & Setup

### Required Python Packages

Install the necessary packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn librosa tensorflow keras
```
