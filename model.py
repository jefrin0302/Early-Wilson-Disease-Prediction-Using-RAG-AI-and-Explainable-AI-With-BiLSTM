#!/usr/bin/env python3
"""
model.py

Full training + saving hybrid model (SVM + LogisticRegression + optional Bi-LSTM),
stacking meta-model, automatic leakage detection & removal,
extensive evaluation graphs, SHAP / permutation importance,
and artifact saving for Wilson_disease_dataset.csv.

Place Wilson_disease_dataset.csv in the same directory and run:
    python model.py

Outputs go to ./wilson_artifacts/
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

# Optional imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, InputLayer
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print("TensorFlow available. Bi-LSTM will be trained.")
except Exception:
    TF_AVAILABLE = False
    print("TensorFlow not available. Bi-LSTM will be skipped. To enable, pip install tensorflow.")

try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available. Will compute SHAP explanations (may be slow).")
except Exception:
    SHAP_AVAILABLE = False
    print("SHAP not available. Will compute permutation importance as fallback.")

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "Wilson_disease_dataset.csv"
OUT_DIR = "wilson_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
TARGET_COL = "Is_Wilson_Disease"

# ----------------------------
# Utility plotting functions
# ----------------------------
def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

def plot_roc_single(y_true, proba, name):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],"k--")
    ax.set_title(f"ROC - {name}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    savefig(fig, f"roc_{name}.png")
    return roc_auc

def plot_pr_single(y_true, proba, name):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_title(f"Precision-Recall - {name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    savefig(fig, f"pr_{name}.png")
    return ap

def plot_confusion_matrix(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    savefig(fig, fname)

def plot_calibration(y_true, prob, name, fname):
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(prob_pred, prob_true, marker='o', label=name)
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration plot: {name}")
    ax.legend()
    savefig(fig, fname)

def plot_history(history, fname_prefix="bilstm_history"):
    if not history:
        return
    hist = history
    if "loss" in hist:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(hist["loss"], label="train_loss")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label="val_loss")
        ax.set_title("Loss")
        ax.legend()
        savefig(fig, f"{fname_prefix}_loss.png")
    if "accuracy" in hist or "acc" in hist:
        # tf might use 'accuracy' or 'acc'
        acc_key = "accuracy" if "accuracy" in hist else "acc"
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(hist[acc_key], label="train_acc")
        if f"val_{acc_key}" in hist:
            ax.plot(hist[f"val_{acc_key}"], label="val_acc")
        ax.set_title("Accuracy")
        ax.legend()
        savefig(fig, f"{fname_prefix}_acc.png")

# ----------------------------
# Data loading & preprocessing with leakage detection + removal
# ----------------------------
def load_df(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Please place 'Wilson_disease_dataset.csv' here.")
    df = pd.read_csv(path)
    print("Loaded dataset:", df.shape)
    return df

def detect_and_remove_leakage(df, target_col=TARGET_COL, verbose=True):
    """
    Detect features that deterministically map to target, or give perfect predictive power alone.
    Returns df with leaking columns removed and a list of removed columns.
    Method:
      - For each column, map unique values -> majority target and compute accuracy.
      - If accuracy == 1.0 (perfect), consider it leaking and remove.
      - Also check numeric correlation abs == 1.0 as an extra check.
    """
    removed = []
    cols = [c for c in df.columns if c != target_col]
    y = df[target_col]
    n = len(df)
    for col in cols:
        ser = df[col]
        # If column is all unique indices (like ID), skip as potential leak if it perfectly matches target (handled below)
        try:
            # Compute mapping accuracy: for each unique value, assign majority label and compute overall accuracy
            mapping = df.groupby(col)[target_col].agg(lambda x: x.mode().iat[0] if len(x.mode())>0 else x.iloc[0])
            # Map original column to predicted target via mapping
            pred = ser.map(mapping)
            acc = (pred == y).mean()
        except Exception:
            acc = 0.0
        # also numeric perfect correlation check
        corr_perfect = False
        if pd.api.types.is_numeric_dtype(ser):
            try:
                corr = df[[col, target_col]].corr().iloc[0,1]
                if pd.notna(corr) and abs(corr) == 1.0:
                    corr_perfect = True
            except Exception:
                corr_perfect = False
        if acc == 1.0 or corr_perfect:
            removed.append(col)
            if verbose:
                print(f"[LEAKAGE] Column '{col}' deterministically predicts the target (accuracy={acc:.3f}, corr_flag={corr_perfect}). Will drop it.")
    # remove found columns
    if removed:
        df = df.drop(columns=removed)
    else:
        if verbose:
            print("No deterministic leakage columns found (accuracy==1.0).")
    return df, removed

def build_preprocessor_and_features(df, target_col=TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    # Detect categorical: those with dtype 'object' or 'category'
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    # Heuristic: sometimes integer-coded categoricals appear numeric. If a numeric column has small nunique relative to dataset, treat as categorical.
    for col in X.columns:
        if col in numeric_features:
            nunique = X[col].nunique(dropna=False)
            if nunique <= 10 and nunique < 0.05 * len(X):
                # treat as categorical
                numeric_features.remove(col)
                categorical_features.append(col)
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    X_proc = preprocessor.fit_transform(X)
    # build feature names
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    except Exception:
        cat_names = categorical_features
    feature_names = numeric_features + cat_names
    return X_proc, y, preprocessor, feature_names

# ----------------------------
# Bi-LSTM builder (optional)
# ----------------------------
def build_bilstm(input_timesteps):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_timesteps, 1)))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ----------------------------
# Main training pipeline
# ----------------------------
def train_and_save():
    # Load
    df = load_df(DATA_PATH)

    # Basic cleanup: drop 'Name' column if present
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])
        print("Dropped 'Name' column.")

    # Confirm target exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV.")

    # Detect & remove leakage
    df_clean, removed_cols = detect_and_remove_leakage(df, TARGET_COL)
    if removed_cols:
        with open(os.path.join(OUT_DIR, "leakage_removed.json"), "w") as f:
            json.dump({"removed_columns": removed_cols}, f, indent=2)
        print("Removed leakage columns saved to leakage_removed.json")

    # Build preprocessor and features
    X_proc, y, preprocessor, feature_names = build_preprocessor_and_features(df_clean, TARGET_COL)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # containers
    test_probas = {}
    test_preds = {}
    model_objects = {}

    # --------------------
    # Optional Bi-LSTM
    # --------------------
    bilstm_history = None
    if TF_AVAILABLE:
        timesteps = X_train.shape[1]
        X_train_lstm = X_train.reshape((X_train.shape[0], timesteps, 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], timesteps, 1))
        print("Training Bi-LSTM...")
        bilstm = build_bilstm(timesteps)
        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        hist = bilstm.fit(X_train_lstm, y_train, validation_split=0.15, epochs=25, batch_size=32, callbacks=[es], verbose=1)
        bilstm_history = hist.history
        proba_bilstm = bilstm.predict(X_test_lstm).ravel()
        pred_bilstm = (proba_bilstm >= 0.5).astype(int)
        test_probas["BiLSTM"] = proba_bilstm
        test_preds["BiLSTM"] = pred_bilstm
        model_objects["BiLSTM"] = bilstm
        # save bilstm model
        bilstm_path = os.path.join(OUT_DIR, "bilstm_model.h5")
        bilstm.save(bilstm_path)
        print("Saved Bi-LSTM to", bilstm_path)
        # Save history plots
        plot_history(bilstm_history, fname_prefix="bilstm_history")
    else:
        print("Skipping Bi-LSTM (TensorFlow not installed).")

    # --------------------
    # SVM
    # --------------------
    print("Training SVM...")
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    proba_svm = svm.predict_proba(X_test)[:,1]
    pred_svm = (proba_svm >= 0.5).astype(int)
    test_probas["SVM"] = proba_svm
    test_preds["SVM"] = pred_svm
    model_objects["SVM"] = svm
    joblib.dump(svm, os.path.join(OUT_DIR, "svm.joblib"))
    print("Saved SVM.")

    # --------------------
    # Base Logistic Regression
    # --------------------
    print("Training base LogisticRegression...")
    logreg = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    proba_log = logreg.predict_proba(X_test)[:,1]
    pred_log = (proba_log >= 0.5).astype(int)
    test_probas["LogReg"] = proba_log
    test_preds["LogReg"] = pred_log
    model_objects["LogReg"] = logreg
    joblib.dump(logreg, os.path.join(OUT_DIR, "logreg_base.joblib"))
    print("Saved LogisticRegression.")

    # --------------------
    # Stacking: build OOF meta features
    # --------------------
    print("Building stacking features (out-of-fold predictions)...")
    base_list = []
    if TF_AVAILABLE:
        base_list.append("BiLSTM")
    base_list += ["SVM", "LogReg"]
    n_models = len(base_list)

    meta_train = np.zeros((X_train.shape[0], n_models))
    meta_test = np.zeros((X_test.shape[0], n_models))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for i, name in enumerate(base_list):
        oof = np.zeros(X_train.shape[0])
        test_fold_preds = np.zeros((kf.get_n_splits(), X_test.shape[0]))
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            if name == "BiLSTM" and TF_AVAILABLE:
                # train small BiLSTM on fold
                timesteps = X_tr.shape[1]
                m = build_bilstm(timesteps)
                m.fit(X_tr.reshape((X_tr.shape[0], timesteps, 1)), y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx], epochs=8, batch_size=32, verbose=0)
                oof[val_idx] = m.predict(X_val.reshape((X_val.shape[0], timesteps, 1))).ravel()
                test_fold_preds[fold_idx, :] = m.predict(X_test.reshape((X_test.shape[0], timesteps, 1))).ravel()
            elif name == "SVM":
                m = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
                m.fit(X_tr, y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx])
                oof[val_idx] = m.predict_proba(X_val)[:,1]
                test_fold_preds[fold_idx, :] = m.predict_proba(X_test)[:,1]
            elif name == "LogReg":
                m = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
                m.fit(X_tr, y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx])
                oof[val_idx] = m.predict_proba(X_val)[:,1]
                test_fold_preds[fold_idx, :] = m.predict_proba(X_test)[:,1]
        meta_train[:, i] = oof
        meta_test[:, i] = test_fold_preds.mean(axis=0)

    # Fit meta model
    meta = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    meta.fit(meta_train, y_train)
    model_objects["meta"] = meta
    joblib.dump(meta, os.path.join(OUT_DIR, "meta_model.joblib"))
    print("Saved meta model.")

    meta_proba = meta.predict_proba(meta_test)[:,1]
    test_probas["StackedMeta"] = meta_proba
    test_preds["StackedMeta"] = (meta_proba >= 0.5).astype(int)

    # --------------------
    # Evaluation & Graphs
    # --------------------
    print("\n--- Evaluation & Graphs ---")
    metrics_summary = {}
    for name in list(test_probas.keys()):
        proba = test_probas[name]
        pred = test_preds[name]
        acc = accuracy_score(y_test, pred)
        try:
            auc_score = roc_auc_score(y_test, proba)
        except Exception:
            auc_score = None
        metrics_summary[name] = {"accuracy": float(acc), "auc": float(auc_score) if auc_score is not None else None}
        print(f"{name}: acc={acc:.4f} auc={auc_score if auc_score is None else f'{auc_score:.4f}'}")

        # Save ROC & PR & confusion & calibration & classification report
        if proba is not None and len(np.unique(y_test)) > 1:
            _ = plot_roc_single(y_test, proba, name)
            _ = plot_pr_single(y_test, proba, name)
            try:
                plot_calibration(y_test, proba, name, fname=f"calibration_{name}.png")
            except Exception as e:
                print("Calibration failed for", name, e)
        # Confusion matrix
        plot_confusion_matrix(y_test, pred, title=f"Confusion Matrix - {name}", fname=f"confusion_{name}.png")
        # classification report
        crep = classification_report(y_test, pred, output_dict=True)
        with open(os.path.join(OUT_DIR, f"classif_report_{name}.json"), "w") as f:
            json.dump(crep, f, indent=2)

    # Combined ROC & PR comparison charts
    try:
        fig, ax = plt.subplots(figsize=(7,6))
        for name, proba in test_probas.items():
            if proba is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        ax.plot([0,1],[0,1],"k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        savefig(fig, "roc_all_models.png")
    except Exception as e:
        print("Combined ROC failed:", e)

    try:
        fig, ax = plt.subplots(figsize=(7,6))
        for name, proba in test_probas.items():
            if proba is None:
                continue
            precision, recall, _ = precision_recall_curve(y_test, proba)
            ap = average_precision_score(y_test, proba)
            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="upper right")
        savefig(fig, "pr_all_models.png")
    except Exception as e:
        print("Combined PR failed:", e)

    # Summary bar chart
    try:
        fig, ax = plt.subplots(figsize=(8,5))
        names = list(metrics_summary.keys())
        accs = [metrics_summary[n]["accuracy"] for n in names]
        aucs = [metrics_summary[n]["auc"] if metrics_summary[n]["auc"] is not None else 0.0 for n in names]
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, accs, width, label="Accuracy")
        ax.bar(x + width/2, aucs, width, label="AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.set_ylim(0,1.05)
        ax.set_title("Model comparison")
        ax.legend()
        savefig(fig, "summary_metrics_comparison.png")
    except Exception as e:
        print("Summary metrics chart failed:", e)

    # --------------------
    # XAI: SHAP or permutation importance on meta level
    # --------------------
    if "meta" in model_objects:
        if SHAP_AVAILABLE:
            try:
                # Explain meta model with KernelExplainer (meta takes meta_test as input)
                # We'll use small background sample (meta_train sample)
                bg = shap.sample(meta_train, min(50, meta_train.shape[0]))
                explainer = shap.KernelExplainer(lambda M: model_objects["meta"].predict_proba(M)[:,1], bg)
                shap_vals = explainer.shap_values(meta_test[:40], nsamples=200)
                # Save simple mean absolute SHAP bar (meta features)
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                labels = base_list
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(range(len(labels)), mean_abs)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
                ax.set_title("SHAP (meta) mean |SHAP value|")
                savefig(fig, "shap_meta_meanabs.png")
                np.save(os.path.join(OUT_DIR, "shap_meta_values.npy"), shap_vals)
                print("Saved SHAP meta explanations.")
            except Exception as e:
                print("SHAP explanation failed:", e)
                # fallback to permutation importance below
                SHAP_FALLBACK = True
        else:
            SHAP_FALLBACK = True

        if not SHAP_AVAILABLE or ('SHAP_FALLBACK' in locals() and SHAP_FALLBACK):
            try:
                perm = permutation_importance(model_objects["meta"], meta_test, y_test, n_repeats=30, random_state=RANDOM_STATE, n_jobs=1)
                importances = perm.importances_mean
                labels = base_list
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(range(len(labels)), importances)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
                ax.set_title("Permutation importance - meta model features")
                savefig(fig, "perm_importance_meta.png")
                with open(os.path.join(OUT_DIR, "perm_importance_meta.json"), "w") as f:
                    json.dump({"features": labels, "importances": importances.tolist()}, f, indent=2)
                print("Saved permutation importance for meta model.")
            except Exception as e:
                print("Permutation importance for meta model failed:", e)

    # --------------------
    # Save preprocessor and other artifacts
    # --------------------
    joblib.dump(preprocessor, os.path.join(OUT_DIR, "preprocessor.joblib"))
    joblib.dump(svm, os.path.join(OUT_DIR, "svm.joblib"))
    joblib.dump(logreg, os.path.join(OUT_DIR, "logreg_base.joblib"))
    joblib.dump(meta, os.path.join(OUT_DIR, "meta_model.joblib"))

    meta_info = {
        "removed_leakage_columns": removed_cols,
        "feature_names": feature_names,
        "model_list": list(test_probas.keys()),
        "metrics_summary": metrics_summary,
        "artifact_path": os.path.abspath(OUT_DIR)
    }
    with open(os.path.join(OUT_DIR, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print("\nAll artifacts, models and plots saved to:", os.path.abspath(OUT_DIR))
    print("Summary metrics:", metrics_summary)
    if removed_cols:
        print("Dropped leakage columns:", removed_cols)
    else:
        print("No deterministic leakage columns were dropped.")

if __name__ == "__main__":
    train_and_save()
