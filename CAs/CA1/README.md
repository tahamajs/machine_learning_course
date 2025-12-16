
# **ML Homework 1 – Regression, Cross-Validation & Logistic Regression**

### **University of Tehran – Machine Learning Course**

## **Overview**

This assignment focuses on  **fundamental concepts in machine learning** , including:

* **Cross-Validation**
* **Regularization (L1 & L2)**
* **Linear Regression (Gradient Descent & Normal Equations)**
* **Maximum Likelihood Estimation**
* **Logistic Regression & Classification Metrics**

The goal is to implement and analyze these models using Python.

---

## **Assignment Structure**

### 📌 **Question 1: Cross-Validation**

* **Concepts:**
  * Purpose of **Cross-Validation** in machine learning.
  * Types of cross-validation ( **k-Fold, Leave-One-Out, etc.** ).
  * Metric-based evaluation of models.
* **Implementation:**
  * Implement  **k-Fold Cross-Validation** .
  * Compare model performance on different validation folds.

### 📌 **Question 2: Regularization in Machine Learning**

* **Concepts:**
  * Difference between **L1 Regularization (Lasso)** and  **L2 Regularization (Ridge)** .
  * Effect of **penalty term (λ)** on model weights.
* **Implementation:**
  * Implement **L1 & L2 regularization** for regression.
  * Compute **closed-form solution** for Ridge Regression.

### 📌 **Question 3: Maximum Likelihood Estimation (MLE)**

* **Concepts:**
  * Deriving the **log-likelihood function** for linear regression.
  * Finding **MLE estimates** by minimizing squared error.
* **Implementation:**
  * Compute  **log-likelihood function** .
  * Optimize parameters using  **MLE** .

### 📌 **Question 4: Exponential Regression Model**

* **Concepts:**
  * Regression of the form:
    y=ewxy = e^{wx}
  * Optimization using  **gradient descent** .
* **Implementation:**
  * Compute **loss function** for exponential regression.
  * Derive  **gradient descent update rule** .

### 📌 **Question 5: Linear Regression - Gradient Descent vs Normal Equations**

* **Concepts:**
  * Compare **Gradient Descent** vs **Normal Equations** for  **linear regression** .
  * Effect of **feature scaling** on convergence speed.
* **Implementation:**
  * Implement **gradient descent** from scratch.
  * Implement **Normal Equation** solution.
  * Compute **Mean Squared Error (MSE)** for both methods.

### 📌 **Question 6: Logistic Regression for Diabetes Classification**

* **Concepts:**
  * Logistic regression for binary classification.
  * Evaluating model using:
    * **Confusion Matrix**
    * **Accuracy**
    * **Precision & Recall**
* **Implementation:**
  * Perform  **Exploratory Data Analysis (EDA)** .
  * Train a  **Logistic Regression classifier** .
  * Compute  **confusion matrix & performance metrics** .


---

## **Evaluation Metrics**

* **Regression Models:**
  * **Mean Squared Error (MSE)**
  * **Root Mean Squared Error (RMSE)**
* **Classification Models:**
  * **Confusion Matrix**
  * **Accuracy**
  * **Precision & Recall**
  * **F1-Score**

---

## **Results Summary**

* **Cross-Validation improves model generalization** .
* **L1 Regularization (Lasso) leads to sparse features, L2 (Ridge) shrinks coefficients** .
* **MLE and Least Squares give equivalent solutions for linear regression** .
* **Gradient Descent requires tuning of learning rate, Normal Equation is faster for small datasets** .
* **Logistic Regression performs well for diabetes classification, with accuracy analysis** .

---

## **References**

* **Pattern Recognition and Machine Learning – Bishop**
* **The Elements of Statistical Learning – Hastie, Tibshirani, Friedman**
* **Andrew Ng’s Machine Learning Course (Stanford CS229)**
