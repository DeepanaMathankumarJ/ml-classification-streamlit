"""
app.py - Streamlit app: upload TEST csv, pick a model, see metrics + confusion matrix.
Models are loaded from /model (no retraining).
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Classifier Lab", page_icon="🧪", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}
    enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    cfg = joblib.load(os.path.join(MODEL_DIR, "config.joblib"))
    return models, enc, cfg


def compute_metrics(model, X, y_true, n_classes):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted",
                                labels=list(range(n_classes)))
    except ValueError:
        auc = float("nan")   # happens if uploaded file contains only one class
    return y_pred, {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ----------------------------- UI -----------------------------
st.title("🧪 Classifier Lab")
st.caption("Upload test data, choose a trained model, and inspect how it performs. "
           "Models are pre-trained - nothing is retrained here.")

try:
    models, label_enc, cfg = load_artifacts()
except Exception as e:
    st.error(f"Could not load models from '{MODEL_DIR}/': {e}")
    st.stop()

target_col = cfg["target_col"]
n_classes = len(label_enc.classes_)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    st.markdown(f"**Target column expected:** `{target_col}`")

if uploaded is None:
    st.info("👈 Upload the test CSV (e.g. `test_data.csv` from the repo) to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if target_col not in data.columns:
    st.error(f"Uploaded file must contain the target column '{target_col}'.")
    st.stop()

unknown = set(data[target_col].unique()) - set(label_enc.classes_)
if unknown:
    st.error(f"Target contains labels the model has never seen: {unknown}")
    st.stop()

X = data.drop(columns=[target_col])
y_true = label_enc.transform(data[target_col])

tab1, tab2, tab3 = st.tabs(["📊 Selected model", "🏁 All models", "🔍 Data preview"])

with tab1:
    model = models[model_name]
    try:
        y_pred, m = compute_metrics(model, X, y_true, n_classes)
    except Exception as e:
        st.error(f"Prediction failed - check that the CSV has the same feature columns "
                 f"as the training data.\n\n{e}")
        st.stop()

    st.subheader(f"{model_name} - evaluation metrics")
    cols = st.columns(6)
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, labels=range(n_classes),
                                    target_names=[str(c) for c in label_enc.classes_],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))

with tab2:
    st.subheader("Every model on the uploaded data")
    rows = []
    for n, mdl in models.items():
        _, mm = compute_metrics(mdl, X, y_true, n_classes)
        rows.append({"ML Model Name": n, **mm})
    comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"))
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])

with tab3:
    st.write(f"{data.shape[0]} rows x {data.shape[1]} columns")
    st.dataframe(data.head(50))
