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

# Sidebar Configuration
st.sidebar.header("TEST DATA")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
st.sidebar.caption("200MB per file • CSV")

# Option to use a bundled default dataset if available
use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)

st.sidebar.markdown("---")
st.sidebar.header("MODEL")
model_name = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Model details card in sidebar
model_descriptions = {
    "Logistic Regression": "Linear model used for binary and multi-class classification via logistic function.",
    "Decision Tree": "Non-linear tree-structured model splitting data based on feature impurity.",
    "KNN": "Instance-based learning finding the closest k training examples in feature space.",
    "Naive Bayes": "Probabilistic classifier based on Bayes' theorem with strong feature independence assumptions.",
    "Support Vector Machine": "Finds an optimal hyperplane maximizing the margin between classification classes."
}
st.sidebar.info(model_descriptions.get(model_name, ""))

# Map dropdown to saved model files
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Load data source
df = None
source_name = "None"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
elif use_bundled:
    try:
        df = pd.read_csv("test_data.csv")
        source_name = "test_data.csv"
    except FileNotFoundError:
        pass

if df is not None:
    num_rows = df.shape[0]
    num_features = df.shape[1] - 1  # excluding target
    
    # Top info pill bar matching your reference screenshot layout
    col_p1, col_p2, col_p3, col_p4 = st.columns([1.2, 1, 1, 1])
    col_p1.markdown(f"📦 **Source**: `{source_name}`")
    col_p2.markdown(f"📊 **Rows**: `{num_rows}`")
    col_p3.markdown(f"⚙️ **Features**: `{num_features}`")
    target_classes = df['target'].nunique() if 'target' in df.columns else 0
    col_p4.markdown(f"🏷️ **Classes**: `{target_classes}`")
    
    st.markdown("---")

    # Tabs layout
    tab_eval, tab_compare, tab_data = st.tabs(["Evaluation", "Model Comparison", "Data"])

    if 'target' not in df.columns:
        st.error("Error: The dataset must contain a 'target' column!")
    else:
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        # Load model safely
        try:
            model_path = model_files[model_name]
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Could not load model file for {model_name}. Ensure it's saved in the `model/` folder. Error: {e}")
            st.stop()

        # Predictions
        y_pred = model.predict(X_test)
        
        # Handle probability / AUC safely for binary vs multi-class
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                auc_score = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        else:
            auc_score = 0.0

        # Metrics computation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        correct_count = (y_pred == y_test).sum()

        with tab_eval:
            st.markdown(f"### **{model_name}** classified **{correct_count}** of **{num_rows}** samples correctly (**{acc*100:.1f}%**).")
            
            # Metric Cards Row
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("ACCURACY", f"{acc:.4f}")
            m2.metric("AUC", f"{auc_score:.4f}")
            m3.metric("PRECISION", f"{prec:.4f}")
            m4.metric("RECALL", f"{rec:.4f}")
            m5.metric("F1 SCORE", f"{f1:.4f}")
            m6.metric("MCC", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix & Classification Report Layout
            col_cm, col_rep = st.columns([1.2, 1])
            
            with col_cm:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                st.pyplot(fig)
                
            with col_rep:
                st.subheader("Per-Class Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        with tab_compare:
            st.subheader("All Models Performance Comparison")
            comparison_data = []
            
            for name, path in model_files.items():
                try:
                    m = joblib.load(path)
                    preds = m.predict(X_test)
                    probs = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
                    if probs is not None:
                        ascore = roc_auc_score(y_test, probs[:, 1] if probs.shape[1]==2 else probs, multi_class='ovr', average='macro')
                    else:
                        ascore = 0.0
                        
                    comparison_data.append({
                        "Model Name": name,
                        "Accuracy": round(accuracy_score(y_test, preds), 4),
                        "AUC": round(ascore, 4),
                        "Precision": round(precision_score(y_test, preds, average='weighted', zero_division=0), 4),
                        "Recall": round(recall_score(y_test, preds, average='weighted', zero_division=0), 4),
                        "F1 Score": round(f1_score(y_test, preds, average='weighted', zero_division=0), 4),
                        "MCC": round(matthews_corrcoef(y_test, preds), 4)
                    })
                except Exception:
                    pass
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)

        with tab_data:
            st.subheader("Dataset Explorer")
            st.dataframe(df, use_container_width=True)
else:
    st.warning("Please upload a test CSV dataset or enable bundled test data via the sidebar to initialize the dashboard.")