"""
train_models.py
Trains 5 classifiers on ONE dataset, evaluates them, and saves everything the
Streamlit app needs (models, test_data.csv, metrics table).

Run:  python train_models.py
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ----------------------------------------------------------------------
# 1. CONFIG  -> change these two lines for YOUR dataset
# ----------------------------------------------------------------------
DATA_PATH = "dataset.csv"      # full dataset downloaded from Kaggle / UCI
TARGET_COL = "target"          # name of the label column in that CSV
# ----------------------------------------------------------------------

RANDOM_STATE = 42
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- load data ----------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Class distribution:\n", df[TARGET_COL].value_counts(), "\n")

X = df.drop(columns=[TARGET_COL])
y_raw = df[TARGET_COL]

# encode labels (works for text labels or numbers)
label_enc = LabelEncoder()
y = label_enc.fit_transform(y_raw)
n_classes = len(label_enc.classes_)

# ---------- train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------- preprocessing ----------
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), cat_cols),
])

# ---------- the five models ----------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

# binary -> 'binary' averaging ; multi-class -> 'weighted'
avg = "binary" if n_classes == 2 else "weighted"

results = []
for name, clf in classifiers.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
    print(f"saved {fname}")

# ---------- save artifacts ----------
joblib.dump(label_enc, os.path.join(MODEL_DIR, "label_encoder.joblib"))

# test_data.csv keeps the ORIGINAL target labels (this is what you upload in the app)
test_df = X_test.copy()
test_df[TARGET_COL] = label_enc.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

joblib.dump({"target_col": TARGET_COL}, os.path.join(MODEL_DIR, "config.joblib"))

metrics_df = pd.DataFrame(results).round(4)
metrics_df.to_csv("metrics_comparison.csv", index=False)
print("\n=== Comparison table (copy into README) ===")
print(metrics_df.to_markdown(index=False))
