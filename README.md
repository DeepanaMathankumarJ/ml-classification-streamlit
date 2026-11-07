# Machine Learning Assignment 2 - Classification & Streamlit Deployment

**Course**: M.Tech (AIML) - Machine Learning  
**Institution**: BITS Pilani

---

## a. Problem Statement
The objective of this assignment is to develop, evaluate, and compare five distinct Machine Learning classification algorithms on a benchmark dataset. The evaluation is conducted across six performance metrics: 
- **Accuracy**
- **Area Under ROC Curve (AUC)**
- **Precision**
- **Recall**
- **F1-Score**
- **Matthews Correlation Coefficient (MCC)**

The final trained models are serialized and integrated into an interactive web interface built with Streamlit, which is deployed live on Streamlit Community Cloud without retraining on user-uploaded test data.

---

## b. Dataset Description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: `sklearn.datasets` (UCI Machine Learning Repository)
- **Problem Type**: Binary Classification (0 = Malignant, 1 = Benign)
- **Total Features**: 30 numeric, continuous features computed from digitized images of fine needle aspirates (FNA) of breast masses (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry).
- **Total Instances**: 569 instances (meets requirement of >500 instances and >12 features).
- **Split Ratio**: 80% Training Data, 20% Hold-out Test Data (`test_data.csv`).

---

## c. GitHub Repository Link
https://github.com/DeepanaMathankumarJ/ml-classification-streamlit.git

## d. Models Used & Evaluation Comparison Table

All models were evaluated on the exact same hold-out test dataset (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | **0.9649** | **0.9954** | **0.9595** | **0.9861** | **0.9726** | **0.9245** | 
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 | 
| **K-Nearest Neighbors (KNN)** | 0.9123 | 0.9559 | 0.9429 | 0.9167 | 0.9296 | 0.8139 | 
| **Naive Bayes (Gaussian)** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 | 
| **Support Vector Machine (SVM)** | 0.9123 | 0.9696 | 0.9189 | 0.9444 | 0.9315 | 0.8102 | 

## e. Observations on Model Performance

| ML Model Name | Observation about model performance | 
| ----- | ----- | 
| **Logistic Regression** | Showed outstanding overall performance, achieving the highest Accuracy (0.9649), AUC (0.9954), Recall (0.9861), and MCC (0.9245). Its linear decision boundary fits the high-dimensional feature space cleanly without overfitting. | 
| **Decision Tree** | Achieved lower overall performance with an Accuracy of 0.9123 and AUC of 0.9157. Non-ensemble decision trees tend to overfit the training split and create sharp axis-aligned boundaries. | 
| **KNN** | Delivered stable results (Accuracy: 0.9123, F1: 0.9296), but suffered slightly due to distance metrics being affected by feature scale variations in unscaled dimensions. | 
| **Naive Bayes** | Displayed strong probabilistic predictions with an excellent AUC score (0.9878) and F1 score (0.9517), benefiting from its conditional independence assumption. | 
| **Support Vector Machine** | Delivered solid sensitivity with an AUC of 0.9696, though its accuracy (0.9123) and MCC (0.8102) were slightly constrained without explicit feature scaling. | 
| **Overall Winner** | **Logistic Regression** is the clear winner for this dataset, leading across virtually every metric, particularly MCC (0.9245) and AUC (0.9954). | 

## Project Folder Structure

```
.
├── app.py                  # Streamlit deployment app logic
├── test_data.csv           # Test dataset used for verification
├── requirements.txt        # Package dependencies for cloud deployment
├── README.md               # Assignment documentation and comparison table
└── model/                  # Directory containing serialized model artifacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib

```