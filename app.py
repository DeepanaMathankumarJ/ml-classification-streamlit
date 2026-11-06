import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.title("ML Model Evaluation & Deployment Dashboard")
st.write("BITS Pilani M.Tech AIML - Machine Learning Assignment 2")

# Sidebar model selection and file upload
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Map dropdown to saved file paths
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.subheader("Test Dataset Preview")
    st.dataframe(test_data.head())
    
    if 'target' not in test_data.columns:
        st.error("Uploaded CSV must contain the 'target' column!")
    else:
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # Load pre-trained model
        model_path = model_files[model_name]
        model = joblib.load(model_path)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.subheader(f"Evaluation Metrics for: {model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("Please upload your `test_data.csv` file using the sidebar to view evaluation results.")