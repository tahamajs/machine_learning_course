### **Project Overview**

- **Institution:** University of Tehran, College of Engineering (ECE Department)
- **Course:** Machine Learning
- **Instructors:** Dr. Abolghasemi, Dr. Arabi, Dr. Tavassolipour
- **Semester:** Fall 1401 (2022)
- **Project Title:** Traditional Persian Music Classification and Clustering

### **1. Introduction & Objectives**

The goal is to perform a real-world Machine Learning project from "Zero to One Hundred." Unlike typical academic projects where clean data is provided, you must handle the entire pipeline:

1. **Data Collection:** Gathering raw audio.
2. **Preprocessing:** Cleaning and preparing data.
3. **Feature Extraction:** Converting audio to vectors.
4. **Modeling:** Classification and Clustering.
5. **Reporting:** Technical analysis.

### **2. Project Definition**

**Core Task:** Classify and cluster Persian Music "Dastgahs" based on instrument data using Machine Learning (and optionally Deep Learning like RNNs/LSTMs).

#### **2.1 Data Collection (Individual Task)**

- **Requirement:** Each student must collect **35 pieces** of music total.
  - 5 pieces for _each_ of the 7 Dastgahs.
- **Uniqueness:** You must check a shared Google Sheet to ensure your selected piece has not already been uploaded by another student.
- **Format:**
  - File type: `.mp3`
  - Length: Between **20 seconds** and **600 seconds** (10 minutes).
  - Must be cut precisely.
- **Deadline:** End of **Dey 16th** (approx. Jan 6th).

ID Mappings for Data Labeling:

| Instrument | ID | | Dastgah | ID |

| :--- | :--- | :--- | :--- | :--- |

| Tar | 0 | | Shur | 0 |

| Kamancheh | 1 | | Segah | 1 |

| Santoor | 2 | | Mahour | 2 |

| Setar | 3 | | Homayoun | 3 |

| Ney | 4 | | Rast-Panjgah | 4 |

| Combination/Other | 5 | | Nava | 5 |

| | | | Chahargah | 6 |

#### **2.2 Initial Report (Group Task)**

- **Content:** A general research report (min. 2 pages) regarding the challenges of classifying Traditional Persian Music.
- **Tone:** Non-technical (readable by a general audience), focusing on the nature of the data.
- **Deadline:** End of **Dey 23rd** (approx. Jan 13th).

#### **2.3 Grouping**

- **Size:** Individual or Groups of max **4 members** .
- **Deadline:** Submit names by **Dey 9th** (approx. Dec 30th).
- _Note:_ Even in groups, data collection quotas are per individual.

---

### **3. Technical Implementation**

#### **3.1 Data Cleaning & Feature Extraction**

- Clean the raw data collected.
- Extract features from the audio signals (Time-domain and Frequency-domain).
- Visualize these features to gain intuition about the data structure.

#### **3.2 Classification**

- **Data Split:** Divide data into Training and Test sets (Test set must be at least **25%** ).
  - _Constraint:_ Ensure proper separation (pay attention to scenarios such as unseen classes/variations in the test set).
- **Models:** You must implement at least **3 distinct methods** and compare them.
  - _Suggested Algorithms:_ Logistic Regression, KNN, SVM, AdaBoost, XGBoost.
  - _Advanced/Optional:_ MLP (compare with/without normalization), Dimensionality Reduction (PCA, LDA, Forward/Backward Selection), Ensemble Learning.
  - _Deep Learning:_ RNN/LSTM (Sequential data) is highly recommended.
- **Analysis:** Analyze accuracy, bias, and overfitting. Plot Error and Accuracy charts.

#### **3.3 Clustering**

- **Methods:** Implement at least **2 distinct clustering methods** .
- **Cluster Counts:** Perform clustering for **k=7** (natural number of Dastgahs) and **k=20** .
- **Analysis:** Analyze intra-cluster similarity vs. inter-cluster differences. Explain why certain data points ended up in specific clusters.

---

### **4. Grading Scheme (Total: 100)**

| **Section**     | **Points** |
| --------------- | ---------- |
| Group Formation | 5          |
| Data Collection | 15         |
| Preprocessing   | 20         |
| Clustering      | 20         |
| Classification  | 20         |
| Final Report    | 20         |

---

### **5. Final Report Requirements**

- **Deadline:** End of **Bahman 14th** (approx. Feb 3rd).
- **Content:**
  - Include the Initial Report as the Introduction.
  - Detailed explanation of preprocessing methods.
  - Analysis of Clustering (k=7 vs k=20).
  - Justification for Classification model selection.
  - Metrics: **Error, F1-Score, Recall, Precision** for every model.
  - Comparison of results.
- **Bonus:** **5% bonus** if the report is written in **LaTeX** .

### **6. Important Policies**

- **Plagiarism:** Zero tolerance for copying code or translating reports word-for-word. Citations are allowed.
- **Support:** Ask questions in the course forum or Telegram group.
- **Contact Emails:**
  - emami.nika@gmail.com
  - mohammad.nili@ut.ac.ir
  - abbasnosrat@gmail.com
  - parisatavana9@gmail.com
