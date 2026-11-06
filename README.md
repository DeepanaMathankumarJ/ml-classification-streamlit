# Machine Learning Assignment 2 - Classification & Streamlit Deployment

**Course**: M.Tech (AIML) - Machine Learning  
**Institution**: BITS Pilani (Work Integrated Learning Programmes Division)  

---

## a. Problem Statement
The objective of this assignment is to develop, evaluate, and compare five distinct Machine Learning classification algorithms on a benchmark dataset. The evaluation is conducted across six performance metrics: Accuracy, Area Under ROC Curve (AUC), Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC). The final trained models are serialized and integrated into an interactive web interface built with Streamlit, which is deployed live on Streamlit Community Cloud without retraining on user-uploaded test data.

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