# Machine Learning Assignment 2 - Classification & Streamlit Deployment

**Course**: M.Tech (AIML) - Machine Learning  
**Developor**: Deepana Jayachandran  

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
- **Dataset Name**: Statlog (Australian Credit Approval) Dataset
- **Source**: OpenML / UCI Machine Learning Repository (`fetch_openml(name='credit-approval', version=1)`)
- **Problem Type**: Binary Classification (`0` = Credit Rejected, `1` = Credit Approved)
- **Total Features**: 15 features (including categorical attributes one-hot encoded into a clean, aligned feature matrix).
- **Total Instances**: 653 instances after handling missing values (meets requirement of >500 instances and >12 features).
- **Split Ratio**: 80% Training Data (522 samples), 20% Hold-out Test Data (131 samples, saved as `test_data.csv`).

---

## c. GitHub Repository Link
https://github.com/DeepanaMathankumarJ/ml-classification-streamlit.git

---

## d. Models Used & Evaluation Comparison Table

All models were evaluated on the exact same hold-out test dataset (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | 0.8092 | 0.8802 | 0.8219 | 0.8333 | 0.8276 | 0.6140 | 
| **Decision Tree** | **0.8473** | 0.8630 | 0.8250 | **0.9167** | **0.8684** | **0.6932** | 
| **K-Nearest Neighbors (KNN)** | 0.7939 | 0.8622 | 0.7711 | 0.8889 | 0.8258 | 0.5853 | 
| **Naive Bayes (Gaussian)** | 0.7863 | 0.8804 | 0.7500 | **0.9167** | 0.8250 | 0.5762 | 
| **Support Vector Machine (SVM)** | 0.8015 | **0.8835** | **0.8194** | 0.8194 | 0.8194 | 0.5991 | 

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance | 
| ----- | ----- | 
| **Logistic Regression** | Performed strongly with balanced metrics (Accuracy: 0.8092, AUC: 0.8802), effectively capturing linear feature relationships after standard scaling. | 
| **Decision Tree** | Achieved the highest overall **Accuracy (0.8473)**, **F1-Score (0.8684)**, and **MCC (0.6932)**. Applying a restricted max depth (`max_depth=5`) successfully prevented overfitting and produced robust splits across mixed feature types. | 
| **KNN** | Delivered stable classification results (Accuracy: 0.7939, AUC: 0.8622), with tuned $k=7$ neighbors proving effective post-scaling. | 
| **Naive Bayes** | Demonstrated exceptional sensitivity with a joint-top **Recall of 0.9167** and strong **AUC of 0.8804**, though independent feature assumptions slightly impacted precision (0.7500). | 
| **Support Vector Machine** | Delivered the top **AUC score (0.8835)** and strong precision (0.8194). Calibrating probabilities via `CalibratedClassifierCV` allowed precise ROC boundary evaluation. | 
| **Overall Winner** | **Decision Tree** is the overall top performer for this dataset, leading across Accuracy (0.8473), F1 (0.8684), and MCC (0.6932). | 

---

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
    ├── support_vector_machine.joblib
    └── scaler.joblib
```