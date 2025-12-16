# Decision_Trees_Clustering_and_Classification_HW4

# **Machine Learning Homework 4**

## **Decision Trees, Clustering, and Classification**

### **Course:** Machine Learning

### **University:** University of Tehran

### **Department:** Electrical and Computer Engineering

### **Professor:** [Insert Professor’s Name]

### **Semester:** Fall 1403

---

## **Overview**

This assignment focuses on  **decision tree classifiers, clustering methods, and spam classification** . You will:

* Implement a **Decision Tree classifier** for spam email classification.
* Explore the **Bias-Variance tradeoff** in tree-based models.
* Analyze  **clustering techniques** , including  **DBSCAN and OPTICS** .
* Implement **K-Means clustering** for customer segmentation.

---

## **Assignment Structure**

The homework consists of  **five main questions** , covering both theoretical and practical aspects.

### **📌 Question 1: Decision Tree for Spam Classification (20 points)**

* Implement a **Decision Tree classifier** using  **Information Gain** .
* Train the model on **email data** to classify emails as  **spam or ham** .
* Use **3-level depth pruning** to improve generalization.
* Apply the trained model on a **test dataset** and predict email categories.

---

### **📌 Question 2: Bias-Variance Tradeoff in Decision Trees (20 points)**

* Compare  **Decision Trees and Bagging Methods** .
* Discuss **bias and variance tradeoffs** in different learning algorithms.
* Analyze how **model complexity** affects  **overfitting and underfitting** .

---

### **📌 Question 3: Clustering Methods – Density-Based Models (20 points)**

* Understand the role of **clustering** in unsupervised learning.
* Investigate **density-based clustering** using  **DBSCAN** .
* Compare **DBSCAN vs. OPTICS** for identifying non-linearly separable clusters.

---

### **📌 Question 4: Decision Tree for Restaurant Wait Time Prediction (25 points)**

* Train a **decision tree model** to predict if a customer will wait at a restaurant.
* Use **entropy and Gini impurity** for splitting.
* Implement the tree  **with and without pre-built libraries** .
* Compare results using different **feature selection** techniques.

---

### **📌 Question 5: K-Means Clustering for Customer Segmentation (25 points)**

* Perform **K-Means clustering** on customer transaction data.
* Analyze clusters based on:
  * **Annual income**
  * **Spending score**
  * **Gender & age distribution**
* Determine the **optimal number of clusters** using  **elbow method** .
* Visualize clusters in  **2D and 3D plots** .

---

## **Implementation Details**

### **📂 Folder Structure**

```
ML_HW4/
│── data/                     # Dataset files
│── models/                   # Trained models
│── notebooks/                # Jupyter notebooks
│── src/                      # Python scripts
│   ├── decision_tree.py      # Decision Tree implementation
│   ├── clustering.py         # DBSCAN, OPTICS, K-Means
│   ├── preprocessing.py      # Data preprocessing functions
│── results/                  # Visualizations and reports
│── requirements.txt          # Dependencies
│── README.md                 # This README file
```

---

## **How to Run the Code**

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repository/ml_hw4.git
   cd ml_hw4
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Decision Tree for Spam Classification**
   ```bash
   python src/decision_tree.py
   ```
4. **Run Clustering Algorithms**
   ```bash
   python src/clustering.py
   ```
5. **Open Jupyter Notebook for Analysis**
   ```bash
   jupyter notebook
   ```

---

|  |  |
| - | - |
