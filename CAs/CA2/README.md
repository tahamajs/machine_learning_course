# **ML Homework 2 – Bayesian Classification & Gaussian Models**

### **University of Tehran – Machine Learning Course**

## **Overview**

This assignment focuses on  **Bayesian classification** ,  **Gaussian distributions** , and  **decision boundaries** . You will:

* Implement  **Naïve Bayes Classification** .
* Compute **decision boundaries** for binary classification problems.
* Analyze **Bayes risk** and  **discriminant functions** .
* Simulate **Gaussian-distributed datasets** and evaluate classifiers.

---

## **Assignment Structure**

### 📌 **Question 1: Decision Boundary for Gaussian Distributions**

* **Concepts:**
  * **Bayes Theorem** for classification.
  * **Likelihood functions** of normal distributions.
  * Finding the **decision boundary** by setting the class conditional probabilities equal.
* **Implementation:**
  * Compute the  **log-likelihood ratio** .
  * Solve for  **decision boundary** .

### 📌 **Question 2: Naïve Bayes Classification**

* **Concepts:**
  * **Naïve Bayes Classifier** assumes feature independence.
  * Computes **posterior probabilities** using:
    P(Y∣X)=P(X∣Y)P(Y)P(X)P(Y | X) = \frac{P(X | Y) P(Y)}{P(X)}
* **Implementation:**
  * Compute  **prior probabilities** .
  * Compute **conditional probabilities** for each feature.
  * Implement  **classification rules** .

### 📌 **Question 3: Bayes Risk Minimization**

* **Concepts:**
  * Minimizing classification risk based on a  **loss function** .
  * **Effect of changing priors** on decision boundary.
* **Implementation:**
  * Compute  **Bayesian risk** .
  * Analyze how **decision boundary shifts** when priors change.

### 📌 **Question 4: Bayesian Decision Boundary for Multivariate Gaussian Distributions**

* **Concepts:**
  * **Multivariate Gaussian Distributions** .
  * Computing the **quadratic decision boundary** for two classes.
* **Implementation:**
  * Compute **mean vectors** and  **covariance matrices** .
  * Compute the  **Bayes decision boundary equation** .

### 📌 **Question 5: Gaussian Classification Simulation**

* **Concepts:**
  * Generate  **2-class Gaussian distributed data** .
  * Compute  **Bayes optimal decision boundary** .
* **Implementation:**
  * Generate  **synthetic Gaussian data** .
  * Compute  **discriminant functions** .
  * Visualize  **decision regions** .

### 📌 **Question 6: Quadratic Optimization with Lagrange Multipliers**

* **Concepts:**
  * Solve a  **constrained quadratic optimization problem** .
  * **Lagrangian formulation** for constrained minimization.
* **Implementation:**
  * Formulate  **Lagrange function** .
  * Solve for  **optimal parameters** .

---

## **Evaluation Metrics**

* **Accuracy** : TP+TNTotal\frac{TP + TN}{Total}
* **Precision** : TPTP+FP\frac{TP}{TP + FP}
* **Recall** : TPTP+FN\frac{TP}{TP + FN}
* **F1-Score** : 2×Precision×RecallPrecision+Recall2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
* **AUROC (Area Under ROC Curve)** .

---

## **Results Summary**

* **Bayes decision boundary** is optimal for minimizing error.
* **Naïve Bayes performs well** when feature independence holds.
* **Gaussian distributions** allow analytical decision boundaries.
* **Changing priors shifts decision boundaries** .

---

## **References**

* **Pattern Recognition and Machine Learning – Bishop**
* **The Elements of Statistical Learning – Hastie, Tibshirani, Friedman**
* **MIT 6.867: Machine Learning Course Notes**

---

## **Authors**

* Mohammad Taha Majlesi

---
