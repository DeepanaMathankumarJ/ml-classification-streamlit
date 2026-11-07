import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)

# =====================================================================
# PROJECT CONFIG
# =====================================================================
TARGET_COL = "target"
CLASS_NAMES = {0: "Rejected (Class 0)", 1: "Approved (Class 1)"}

PROJECT_TITLE = "Australian Credit Approval"
PROJECT_TAGLINE = "Comparing five machine-learning classifiers on the Statlog Australian Credit dataset"
STUDENT_NAME = "Deepana Mathankumar J"
STUDENT_ID = "2026AK05687"

DATASET = {
    "Name": "Statlog (Australian Credit Approval)",
    "Source": "UCI Machine Learning Repository (via OpenML)",
    "Instances": 690,
    "Features": 14,
    "Task": "Binary classification",
    "Class balance": "383 Rejected (55.5%)  |  307 Approved (44.5%)",
    "Train / Test split": "80% / 20%, stratified, random_state = 42",
}
PROBLEM_STATEMENT = (
    "Given a set of financial and demographic attributes for credit card applications, "
    "predict whether a credit application should be **approved** or **rejected**. "
    "Reliable automated classification mitigates financial risk while ensuring fast, "
    "equitable credit decisions for applicants."
)
FEATURE_GROUPS = (
    "Comprises 14 mixed-type attributes (a mix of continuous and nominal values) "
    "capturing personal financial history, customer background metrics, and application "
    "indicators without exposing confidential identifier strings."
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear model passing weighted feature inputs through the logistic function for probability estimation.",
    "Decision Tree": "Non-linear tree structure recursively splitting feature spaces to minimize node impurity.",
    "KNN": "Instance-based learner assigning class labels based on majority vote of nearest training instances.",
    "Naive Bayes": "Probabilistic classifier leveraging Bayes' theorem under strong feature independence assumptions.",
    "Support Vector Machine": "Finds optimal maximum-margin hyperplanes between classes (Calibrated probability pipeline).",
}
MODEL_BADGES = {
    "Logistic Regression": "Linear", "Decision Tree": "Tree-based", "KNN": "Instance-based",
    "Naive Bayes": "Probabilistic", "Support Vector Machine": "Margin-based",
}

# =====================================================================
# PAGE SETUP + CSS
# =====================================================================
st.set_page_config(page_title=f"{PROJECT_TITLE} | ML Dashboard", page_icon="💳", layout="wide")

st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #0f4c81 0%, #6610f2 100%);
    padding: 28px 34px; border-radius: 14px; color: #ffffff; margin-bottom: 18px;
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; color: #ffffff; }
.hero p  { margin: 0; font-size: 1.05rem; opacity: 0.92; color: #ffffff; }
.pill {
    display: inline-block; padding: 3px 12px; margin: 12px 8px 0 0; border-radius: 999px;
    background: rgba(255,255,255,0.18); font-size: 0.82rem; color: #ffffff;
}
.metric-card {
    background-color: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.28);
    padding: 16px; border-radius: 10px; text-align: center;
}
.metric-card h4 { margin: 0 0 4px 0; }
.metric-card hr { margin: 8px 0; border-color: rgba(128,128,128,0.3); }
.fact-row { display:flex; justify-content:space-between; padding:7px 2px;
            border-bottom:1px solid rgba(128,128,128,0.2); font-size:0.95rem; }
.fact-row span:first-child { opacity:0.7; }
.winner {
    border-left: 5px solid #198754; background: rgba(25,135,84,0.10);
    padding: 12px 16px; border-radius: 6px; margin: 8px 0 14px 0;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HELPERS
# =====================================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('model/scaler.joblib')
    except Exception:
        return None

def align_features(model, X):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        return X
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:6]}")
    return X[list(cols)]

def evaluate(model, X, y, model_name):
    Xa = align_features(model, X)
    scaler = load_scaler()
    if scaler is not None and model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        try:
            Xa = scaler.transform(Xa)
        except Exception:
            pass
            
    y_pred = model.predict(Xa)
    y_prob = model.predict_proba(Xa)
    n_classes = y_prob.shape[1]
    avg = "binary" if n_classes == 2 else "weighted"
    
    if n_classes == 2:
        auc = roc_auc_score(y, y_prob[:, 1])
    else:
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="weighted")
        
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return y_pred, y_prob, metrics

def label(v):
    return CLASS_NAMES.get(v, str(v))

# =====================================================================
# HERO HEADER
# =====================================================================
st.markdown(f"""
<div class="hero">
  <h1>💳 {PROJECT_TITLE} - ML Model Evaluation Dashboard</h1>
  <p>{PROJECT_TAGLINE}</p>
  <span class="pill">📚 BITS Pilani - M.Tech (AIML)</span>
  <span class="pill">🧠 Machine Learning - Assignment</span>
  <span class="pill">👤 {STUDENT_NAME} | {STUDENT_ID}</span>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.header("📁 DATASET")
uploaded_file = st.sidebar.file_uploader("Upload test dataset (CSV)", type=["csv"])
use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test data", f, "test_data.csv", "text/csv")
st.sidebar.caption(f"CSV must contain all {DATASET['Features']} feature columns and a `{TARGET_COL}` column.")

st.sidebar.markdown("---")
st.sidebar.header("🤖 MODEL")
model_name = st.sidebar.selectbox("Select classification model", list(MODEL_FILES.keys()))
st.sidebar.info(f"**{MODEL_BADGES[model_name]}** - {MODEL_DESCRIPTIONS[model_name]}")

# =====================================================================
# LOAD DATA
# =====================================================================
test_df, test_source = None, "None"
try:
    if uploaded_file is not None:
        test_df, test_source = pd.read_csv(uploaded_file), uploaded_file.name
    elif use_bundled and os.path.exists("test_data.csv"):
        test_df, test_source = pd.read_csv("test_data.csv"), "test_data.csv (bundled)"
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")

tab_about, tab_eval, tab_compare, tab_data = st.tabs(
    ["🏠 About the Project", "📊 Evaluation", "⚖️ Model Comparison", "📂 Data Explorer"]
)

# ---------------------------------------------------------------------
# TAB 1 - ABOUT
# ---------------------------------------------------------------------
with tab_about:
    st.subheader("🎯 Problem Statement")
    st.markdown(PROBLEM_STATEMENT)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("🗂️ Dataset at a Glance")
        facts = "".join(f'<div class="fact-row"><span>{k}</span><span><b>{v}</b></span></div>'
                        for k, v in DATASET.items())
        st.markdown(facts, unsafe_allow_html=True)
        st.markdown("&nbsp;")
        st.markdown(f"**Target classes:** " + ", ".join(f"`{k}` = {v}" for k, v in CLASS_NAMES.items()))
    with c2:
        st.subheader("🧬 About the Features")
        st.markdown(FEATURE_GROUPS)
        st.subheader("🛠️ Pipeline")
        st.markdown(
            "1. Load dataset & stratified 80/20 split\n"
            "2. Train 5 classifiers on the training set\n"
            "3. Save trained models with `joblib`\n"
            "4. This app loads saved models and scores them on **test data only**"
        )

    st.subheader("🤖 Models Used")
    st.dataframe(pd.DataFrame({
        "Model": list(MODEL_DESCRIPTIONS.keys()),
        "Family": [MODEL_BADGES[m] for m in MODEL_DESCRIPTIONS],
        "How it works": list(MODEL_DESCRIPTIONS.values()),
    }), hide_index=True)

    st.subheader("📏 Metrics Reported")
    st.markdown(
        "**Accuracy** - **AUC** - **Precision** - **Recall** - **F1** - **MCC**."
    )
    st.info("👈 Use the sidebar to choose a model, then open the **Evaluation** tab.")

# ---------------------------------------------------------------------
# DATA TABS HANDLER
# ---------------------------------------------------------------------
def need_data_message():
    st.warning("Upload a test CSV or tick **Use bundled test_data.csv** in the sidebar to see results.")

data_ok = test_df is not None
if data_ok and TARGET_COL not in test_df.columns:
    data_ok = False
    for t in (tab_eval, tab_compare, tab_data):
        with t:
            st.error(f"The uploaded dataset must contain a '{TARGET_COL}' column.")

if data_ok:
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    n_rows, n_feats, n_classes = len(test_df), X_test.shape[1], y_test.nunique()

    try:
        models = {n: load_model(p) for n, p in MODEL_FILES.items()}
        X_test = pd.get_dummies(X_test, drop_first=True)
        align_features(models[model_name], X_test)
    except Exception as e:
        for t in (tab_eval, tab_compare, tab_data):
            with t:
                st.error(f"Error loading models or aligning features: {e}")
        data_ok = False

if not data_ok and test_df is None:
    for t in (tab_eval, tab_compare, tab_data):
        with t:
            need_data_message()

if data_ok:
    # ----------------------------- TAB 2 - EVALUATION -----------------------------
    with tab_eval:
        model = models[model_name]
        y_pred, y_prob, m = evaluate(model, X_test, y_test, model_name)
        correct = int((y_pred == y_test.values).sum())

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            <div class="metric-card"><h4>🧪 Test Dataset</h4>
            <p style="font-size:16px;font-weight:bold;color:#0d6efd;margin:4px 0;">{test_source}</p><hr>
            <p style="margin:4px 0;"><b>Rows:</b> {n_rows} &nbsp;|&nbsp; <b>Features:</b> {n_feats} &nbsp;|&nbsp; <b>Classes:</b> {n_classes}</p>
            </div>""", unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"""
            <div class="metric-card"><h4>⚙️ Active Model</h4>
            <p style="font-size:16px;font-weight:bold;color:#a370f7;margin:4px 0;">{model_name}</p><hr>
            <p style="margin:4px 0;"><b>Family:</b> {MODEL_BADGES[model_name]} &nbsp;|&nbsp; <b>Mode:</b> Pre-trained (joblib)</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f"### **{model_name}** classified **{correct}** of **{n_rows}** test samples correctly (**{m['Accuracy']*100:.1f}%**)")

        cols = st.columns(6)
        for c, (k, v) in zip(cols, m.items()):
            c.metric(k.upper() if k != "F1" else "F1 SCORE", f"{v:.4f}")

        st.markdown("---")
        classes = sorted(set(y_test.unique().tolist()) | set(CLASS_NAMES))
        names = [label(c) for c in classes]
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        left, mid, right = st.columns([1, 1, 1.1])
        with left:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=names, yticklabels=names, annot_kws={"size": 14})
            ax.set_xlabel("Predicted Label"); ax.set_ylabel("Actual Label")
            st.pyplot(fig); plt.close(fig)
        with mid:
            st.subheader("ROC Curve")
            if y_prob.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                fig, ax = plt.subplots(figsize=(4.4, 3.8))
                ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.4f}")
                ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
                ax.fill_between(fpr, tpr, alpha=0.12)
                ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                st.pyplot(fig); plt.close(fig)
            else:
                st.caption("ROC curve is shown for binary problems.")
        with right:
            st.subheader("Per-Class Report")
            rep = classification_report(y_test, y_pred, labels=classes, target_names=names,
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.style.format("{:.4f}"))

        st.markdown("#### 🔎 Predictions")
        out = test_df.copy()
        out["Actual"] = [label(v) for v in y_test]
        out["Predicted"] = [label(v) for v in y_pred]
        out["Confidence"] = y_prob.max(axis=1).round(3)
        out["Correct"] = y_pred == y_test.values
        front = ["Actual", "Predicted", "Confidence", "Correct"]
        out = out[front + [c for c in out.columns if c not in front]]
        wrong = out[~out["Correct"]]
        with st.expander(f"Misclassified samples ({len(wrong)})", expanded=False):
            if len(wrong):
                st.dataframe(wrong)
            else:
                st.success("No misclassified samples - perfect score on this data!")
        st.download_button("⬇️ Download all predictions (CSV)", out.to_csv(index=False),
                           f"predictions_{model_name.lower().replace(' ', '_')}.csv", "text/csv")

    # ----------------------------- TAB 3 - COMPARISON -----------------------------
    with tab_compare:
        st.subheader("All Models - Performance on the Selected Test Data")
        rows = []
        for n, mdl in models.items():
            _, _, mm = evaluate(mdl, X_test, y_test, n)
            rows.append({"ML Model Name": n, **{k: round(v, 4) for k, v in mm.items()}})
        comp = pd.DataFrame(rows).set_index("ML Model Name")

        best = comp["MCC"].idxmax()
        st.markdown(
            f'<div class="winner">🏆 <b>Best model by MCC: {best}</b> '
            f'(MCC {comp.loc[best, "MCC"]:.4f}, AUC {comp.loc[best, "AUC"]:.4f}, '
            f'Accuracy {comp.loc[best, "Accuracy"]:.4f})</div>', unsafe_allow_html=True)

        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#1f7a4d55"))
        st.caption("Green cells mark the best value in each metric column.")

        metric_pick = st.multiselect("Metrics to chart", list(comp.columns),
                                     default=["Accuracy", "AUC", "F1", "MCC"])
        if metric_pick:
            st.bar_chart(comp[metric_pick])
        st.download_button("⬇️ Download comparison table (CSV)", comp.reset_index().to_csv(index=False),
                           "model_comparison.csv", "text/csv")

    # ----------------------------- TAB 4 - DATA EXPLORER -----------------------------
    with tab_data:
        st.subheader("Test Dataset Preview")
        d1, d2, d3 = st.columns(3)
        d1.metric("Rows", n_rows); d2.metric("Features", n_feats); d3.metric("Classes", n_classes)

        left, right = st.columns([1, 1.6])
        with left:
            st.markdown("**Class distribution**")
            dist = y_test.map(label).value_counts()
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.bar(dist.index, dist.values, color=["#0d6efd", "#6610f2", "#fd7e14", "#198754"][:len(dist)])
            for i, v in enumerate(dist.values):
                ax.text(i, v, str(v), ha="center", va="bottom")
            ax.set_ylabel("Samples")
            st.pyplot(fig); plt.close(fig)
        with right:
            st.markdown("**Feature explorer**")
            feat = st.selectbox("Choose a feature", list(X_test.columns))
            fig, ax = plt.subplots(figsize=(6, 3.2))
            for cls in sorted(y_test.unique()):
                ax.hist(X_test.loc[y_test == cls, feat], bins=20, alpha=0.6, label=label(cls))
            ax.set_xlabel(feat); ax.set_ylabel("Count"); ax.legend()
            st.pyplot(fig); plt.close(fig)

        st.markdown("**Raw data**")
        st.dataframe(test_df)
        with st.expander("Summary statistics"):
            st.dataframe(X_test.describe().T.round(3))