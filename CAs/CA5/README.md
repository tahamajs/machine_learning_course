# **Concepts Covered in ML Homework 5**

This assignment covers **fundamental and advanced concepts** in  **neural networks** ,  **MLPs** ,  **CNNs** , and  **deep learning optimization** . Below is a structured breakdown of the key **theoretical concepts** in this homework.

---

## **1. Neural Networks vs. Decision Trees**

* **Comparison of Decision Trees and Neural Networks**
  * Decision Trees: Interpretable, prone to overfitting, works well for structured/tabular data.
  * Neural Networks: Handles complex patterns, requires more data, works well for unstructured data like images/audio.
* **Importance of Non-Linear Activation Functions**
  * Linear functions limit network expressivity.
  * Non-linear activations like  **ReLU** ,  **Sigmoid** , and **Tanh** enable complex decision boundaries.
* **Gradient Descent and Loss Landscapes**
  * **SGD (Stochastic Gradient Descent)** optimizes parameters iteratively.
  * Challenges:  **Vanishing gradients** ,  **local minima** , and **plateaus** in loss surfaces.

---

## **2. Multilayer Perceptron (MLP)**

* **MLP Architecture**
  * Input layer → Hidden layers → Output layer
  * Uses activation functions like  **Sigmoid, ReLU, Softmax** .
* **Forward Propagation**
  * Computes activations layer by layer using:
    z=Wx+bz = W x + b
    a=f(z)a = f(z)
  * **Loss Function** : Measures error (e.g.,  **Cross-Entropy Loss** ,  **MSE** ).
* **Backpropagation (Gradient Calculation)**
  * Uses **chain rule** to propagate errors backward and update weights.
  * Gradients computed for:
    ∂J∂W\frac{\partial J}{\partial W}
    ∂J∂b\frac{\partial J}{\partial b}
  * Weight update:
    W=W−α∇JW = W - \alpha \nabla J

---

## **3. Perceptron & Step Function**

* **Perceptron Model**
  * Binary classifier using a **step function** as activation.
  * Cannot solve non-linearly separable problems (e.g., XOR).
* **XOR Problem and Multi-Layer Perceptron**
  * A single-layer perceptron **fails** to solve XOR.
  * **Adding a hidden layer** with non-linear activation enables solving XOR.

---

## **4. Convolutional Neural Networks (CNNs)**

* **Convolutional Layers**
  * Extract features using  **filters/kernels** .
  * Detect patterns like edges, textures.
* **Translation Invariance**
  * CNNs learn location-invariant features using  **shared weights** .
* **Edge Detection with Filters**
  * Apply **3×3 kernels** for detecting **gradients (edges)** in images.

---

## **5. MLP for Housing Price Prediction**

* **Feature Engineering & Data Preprocessing**
  * Handle  **missing values** .
  * Apply **one-hot encoding** for categorical features.
  * Normalize numerical features.
* **Regression Metrics**
  * **RMSE (Root Mean Squared Error)**
  * **MAE (Mean Absolute Error)**
  * **MAPE (Mean Absolute Percentage Error)**
  * **Log-scaled RMSE**
* **Overfitting vs. Underfitting**
  * **Overfitting** : Too complex model, memorizes data.
  * **Underfitting** : Model is too simple, lacks accuracy.

---

## **6. CNN Architectures: LeNet-5 vs. ResNet**

* **LeNet-5**
  * **Early CNN architecture** for digit recognition.
  * Uses  **convolutional layers, pooling layers, fully connected layers** .
* **Residual Neural Networks (ResNet)**
  * **Solves vanishing gradient problem** using  **skip connections** .
  * **Deeper networks train better** without degradation.
* **Training on MNIST**
  * MLP → 90%+ accuracy.
  * LeNet-5 → 95%+ accuracy.
  * ResNet-18 → 95%+ accuracy.
