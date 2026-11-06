import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling tiles & cards
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("DATASETS")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)

st.sidebar.markdown("---")
st.sidebar.header("MODEL")
model_name = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

model_descriptions = {
    "Logistic Regression": "Linear model used for binary and multi-class classification via logistic function.",
    "Decision Tree": "Non-linear tree-structured model splitting data based on feature impurity.",
    "KNN": "Instance-based learning finding the closest k training examples in feature space.",
    "Naive Bayes": "Probabilistic classifier based on Bayes' theorem with strong feature independence assumptions.",
    "Support Vector Machine": "Finds an optimal hyperplane maximizing the margin between classification classes."
}
st.sidebar.info(model_descriptions.get(model_name, ""))

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Load test data
test_df = None
test_source = "None"

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    test_source = uploaded_file.name
elif use_bundled:
    try:
        test_df = pd.read_csv("test_data.csv")
        test_source = "test_data.csv"
    except FileNotFoundError:
        pass

# Optional: Load train data if saved locally for comparison metrics
train_df = None
try:
    # Assuming train data might be saved or loaded if available
    # We can check a standard filename if you saved it during training phase
    train_df = pd.read_csv("train_data.csv") 
except FileNotFoundError:
    pass

if test_df is not None:
    test_rows = test_df.shape[0]
    test_features = test_df.shape[1] - 1  # excluding target
    test_classes = test_df['target'].nunique() if 'target' in test_df.columns else 0

    # --- TOP METRIC TILES / CARDS FOR DATASET DETAILS ---
    st.markdown("### 📋 Dataset Overview & Statistics")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧪 Test Data Source</h4>
            <p style="font-size: 18px; font-weight: bold; color: #0d6efd;">{test_source}</p>
            <hr style="margin:5px 0;">
            <p><b>Rows:</b> {test_rows} | <b>Features:</b> {test_features}</p>
            <p><b>Target Classes:</b> {test_classes}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        if train_df is not None:
            train_rows = train_df.shape[0]
            train_features = train_df.shape[1] - 1
            train_classes = train_df['target'].nunique() if 'target' in train_df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏋️ Train Data Info</h4>
                <p style="font-size: 18px; font-weight: bold; color: #198754;">Loaded Locally</p>
                <hr style="margin:5px 0;">
                <p><b>Rows:</b> {train_rows} | <b>Features:</b> {train_features}</p>
                <p><b>Target Classes:</b> {train_classes}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏋️ Train Data Info</h4>
                <p style="font-size: 16px; color: #6c757d;">(Train CSV not found in workspace)</p>
                <hr style="margin: