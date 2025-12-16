# **HW3: Optimization Methods in Machine Learning**

### **University of Tehran – Machine Learning Course**

## **Overview**

This assignment focuses on **optimization algorithms** in machine learning, including:

* **Line Search Algorithms**
* **Newton’s Method**
* **Quasi-Newton Methods (BFGS, DFP)**
* **Gradient Descent & Steepest Descent**
* **Hessian Matrix & Eigenvalue Decomposition**

The goal is to understand **mathematical formulations** and implement **optimization techniques** to solve function minimization problems.

---

## **Assignment Structure**

### 📌 **Question 1: Line Search Algorithms**

* **Concepts:**
  * General form of **line search** in optimization.
  * Definition of  **descent direction** .
  * Methods for determining **step length** (Exact Line Search, Inexact Line Search, Backtracking).
  * Mathematical derivation of step length in  **quadratic functions** .
* **Code Implementation:**
  * Compute  **gradient and Hessian matrix** .
  * Apply **line search method** to optimize a given function.

### 📌 **Question 2: Newton’s Method for Optimization**

* **Concepts:**
  * Standard  **Newton’s optimization algorithm** .
  * Conditions where **Newton’s method fails** (ill-conditioned Hessian, singular matrices).
  * Introduction to  **Quasi-Newton Methods (BFGS, DFP)** .
* **Code Implementation:**
  * Implement **Newton’s method** for function minimization.
  * Compare with  **gradient descent** .
  * Use **quasi-Newton (BFGS, DFP)** for optimization.

### 📌 **Question 3: Eigenvalue Decomposition in Optimization**

* **Concepts:**
  * **Eigenvalue decomposition** for square matrices.
  * Importance of  **Hessian matrix analysis** .
  * Application in  **modified Newton’s method** .
* **Code Implementation:**
  * Compute  **eigenvalues & eigenvectors** .
  * Modify  **Newton’s method using eigenvalue decomposition** .

### 📌 **Question 4: Function Optimization using Steepest Descent & Newton’s Method**

* **Concepts:**
  * Optimization of  **Rosenbrock function** .
  * Comparison of  **Steepest Descent vs. Newton’s Method** .
  * Using **function value & distance metrics** to evaluate performance.
* **Code Implementation:**
  * Implement  **Steepest Descent** .
  * Implement  **Newton’s method** .
  * Combine both methods for improved optimization.
  * **Plot function value & distance to optimal point** .

---

## **Evaluation Metrics**

* **Function value vs. Iteration count**
* **Distance to optimal point vs. Iteration count**
* **Comparison of Steepest Descent, Newton’s Method & Hybrid Approach**

---

## **Results Summary**

* **Newton’s method** converges **faster** but requires a well-conditioned Hessian.
* **Steepest Descent** is **robust but slow** due to reliance on first-order derivatives.
* **Hybrid Approach** (combining Newton & GD) provides a  **balance between speed & stability** .

---

## **References**

* **Boyd & Vandenberghe, Convex Optimization**
* **Nocedal & Wright, Numerical Optimization**
* **MIT 6.867: Machine Learning Course Notes**

---

## **Authors**

* Mohammad Taha Majlesi
