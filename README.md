# Artificial Intelligence Assignment 2

## Overview

This project implements and compares two machine learning classification algorithms — **K-Nearest Neighbors (KNN)** and **Decision Tree** — using the **Breast Cancer dataset** provided by the `sklearn` library.

The goal of the project is to evaluate the performance of these models in predicting whether a tumor is **malignant** or **benign** based on numerical features extracted from medical images.

The models are evaluated using common classification metrics including **Accuracy**, **F1-score**, and **Confusion Matrix**. A bar chart visualization is also generated to compare model performance.

---

## Dataset

The project uses the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.

Dataset characteristics:

* **Total samples:** 569
* **Number of features:** 30
* **Target classes:**

  * **Malignant (0)** – cancerous tumor
  * **Benign (1)** – non-cancerous tumor
    
---

## Algorithms Used

### K-Nearest Neighbors (KNN)

KNN is a supervised learning algorithm that classifies a data point based on the majority class among its **k nearest neighbors** in the feature space.

For this project:

* `k = 5`
* Distance metric: Euclidean distance
  
---

### Decision Tree

A Decision Tree is a tree-based model that classifies data by repeatedly splitting the dataset based on feature values. Each internal node represents a decision rule and each leaf node represents a classification result.

---

## Evaluation Metrics

To compare model performance, the following metrics are used:

### Accuracy

Accuracy measures the proportion of correctly classified samples.

Accuracy = (Correct Predictions) / (Total Predictions)

---

### F1-score

The F1-score is the harmonic mean of **precision** and **recall**, providing a balanced evaluation of classification performance.

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install scikit-learn matplotlib
```

---

### 2. Run the program

```bash
python main.py
```

---

### 3. Program Output

The program will:

1. Load the dataset
2. Split the dataset into training and testing sets
3. Train both models
4. Calculate evaluation metrics
5. Display confusion matrices
6. Generate a **bar chart comparing model performance**

The chart will be saved in the **results/** folder.

---

## Example Results

Typical results may look like:

| Model         | Accuracy | F1-score |
| ------------- | -------- | -------- |
| KNN           | 0.96     | 0.96     |
| Decision Tree | 0.94     | 0.94     |

Both models perform well on this dataset, with KNN often achieving slightly higher performance.
