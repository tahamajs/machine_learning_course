
# **Machine Learning Course - Assignments & Final Project**
This repository contains a series of assignments and a final project for the **Machine Learning Course** at the **University of Tehran**. The assignments focus on fundamental and advanced concepts in **machine learning**, including **linear regression, logistic regression, classification, optimization, deep learning, and clustering**.

## 📌 **Table of Contents**
- [Assignments](#assignments)
  - [HW1: Cross-Validation & Regularization](#hw1-cross-validation--regularization)
  - [HW2: Naïve Bayes & Bayesian Decision Theory](#hw2-naïve-bayes--bayesian-decision-theory)
  - [HW3: Optimization Algorithms](#hw3-optimization-algorithms)
  - [HW4: Decision Trees & Clustering](#hw4-decision-trees--clustering)
  - [HW5: Deep Learning & Neural Networks](#hw5-deep-learning--neural-networks)
- [Final Project](#final-project)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Authors](#authors)

---

## **Assignments**
Each homework contains **theoretical questions** and **implementation tasks**, with simulation problems using Python.

### **HW1: Cross-Validation & Regularization**
- **Concepts Covered:**
  - Cross-Validation methods
  - Regularization techniques (L1 & L2)
  - Linear regression using closed-form and gradient descent
- **Tasks:**
  - Implement **k-fold cross-validation** for model selection.
  - Compare **Lasso (L1) and Ridge (L2) Regularization**.
  - Apply gradient descent on a **regression problem**.

### **HW2: Naïve Bayes & Bayesian Decision Theory**
- **Concepts Covered:**
  - Naïve Bayes classifier
  - Bayesian decision theory
  - Discriminant functions
- **Tasks:**
  - Classify binary data using **Naïve Bayes**.
  - Implement **Bayesian decision rule** for Gaussian-distributed classes.
  - Compute and visualize **decision boundaries**.

### **HW3: Optimization Algorithms**
- **Concepts Covered:**
  - Line search optimization
  - Newton’s method
  - Quasi-Newton methods (BFGS, DFP)
- **Tasks:**
  - Implement **gradient descent** for a given function.
  - Compare Newton’s method with gradient-based approaches.
  - Perform eigenvalue decomposition on matrices.

### **HW4: Decision Trees & Clustering**
- **Concepts Covered:**
  - Decision tree learning
  - Bias-variance tradeoff
  - Clustering algorithms (DBSCAN, OPTICS)
- **Tasks:**
  - Implement a **decision tree classifier** using **information gain**.
  - Analyze **bias-variance tradeoff** in machine learning models.
  - Compare **DBSCAN vs. OPTICS clustering**.

### **HW5: Deep Learning & Neural Networks**
- **Concepts Covered:**
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Networks (CNNs)
  - Loss functions & optimization
- **Tasks:**
  - Implement a **fully connected neural network (MLP)**.
  - Design an **XOR classifier** using a perceptron.
  - Train a **CNN** on image classification tasks.

---

## **Final Project**
The final project focuses on **speaker gender classification & voice authentication** using machine learning models.

### **Project Overview**
- **Goal:** Develop a **machine learning model** to classify the **gender of a speaker** and verify their **identity**.
- **Data Processing:**
  - Extract features such as **MFCC, Spectral Centroid, Zero-Crossing Rate**.
  - Perform **data normalization and noise removal**.
  - Implement **feature selection & transformation**.
- **Modeling:**
  - Train **classification models (SVM, MLP, XGBoost)**.
  - Use **clustering techniques** for grouping similar speakers.
  - Evaluate using **ROC curves, Precision, Recall, and F1-score**.
- **Final Deliverables:**
  - **Preprocessed dataset**
  - **Trained models with evaluation metrics**
  - **Report with findings & visualizations**

---

## **Installation & Setup**
To set up the environment, install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn librosa tensorflow keras
```

Ensure you have `Jupyter Notebook` installed if you want to run the notebooks:

```bash
pip install notebook
```

---

## **How to Run**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/machine_learning_course.git
   cd machine_learning_course
   ```

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open the desired assignment and execute the cells.

3. **Run a Python Script**:
   ```bash
   python hw1_cross_validation.py
   ```

---

## **Authors**
- **Instructor:** Dr. E’rabi, Dr. Abolghasemi
- **University:** University of Tehran
- **Course:** Machine Learning - Electrical & Computer Engineering Department
