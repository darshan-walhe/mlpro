import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    MODEL_PATH,
    OUTPUTS_DIR,
    prepare_dataset,
)


def build_pipeline():
    """
    Build a scikit-learn Pipeline:
      1. ColumnTransformer
         • OneHotEncoder for categorical features
         • StandardScaler for numeric features
      2. LogisticRegression with class_weight='balanced'
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the trained pipeline and return a dict of metrics.
    Also generates and saves plots (confusion matrix, ROC curve).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    report = classification_report(y_test, y_pred, target_names=["Not Confirmed", "Confirmed"])

    # ── Confusion Matrix Plot ─────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Confirmed", "Confirmed"],
        yticklabels=["Not Confirmed", "Confirmed"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix - Logistic Regression", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved -> {cm_path}")

    # ── ROC Curve Plot ────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(fpr, tpr, color="#2563eb", linewidth=2,
             label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title("ROC Curve - Logistic Regression", fontsize=14)
    ax2.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
    fig2.savefig(roc_path, dpi=150)
    plt.close(fig2)
    print(f"  [OK] ROC curve saved -> {roc_path}")

    return metrics, report


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("  PNR Waitlist Confirmation - Training Pipeline")
    print("=" * 60)

    # ── Phase 1 & 2: Load + Preprocess ────────────────────────────────────
    print("\n[1/4] Loading and preprocessing dataset ...")
    X, y, df = prepare_dataset()
    print(f"  - Dataset shape: {df.shape}")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Target distribution:\n{y.value_counts().to_string()}")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\n[2/4] Splitting into train/test (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  - Train: {X_train.shape[0]} samples")
    print(f"  - Test : {X_test.shape[0]} samples")

    # ── Phase 3: Train ────────────────────────────────────────────────────
    print("\n[3/4] Training Logistic Regression pipeline ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("  [OK] Training complete")

    # ── Phase 4: Evaluate ─────────────────────────────────────────────────
    print("\n[4/4] Evaluating model ...")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    metrics, report = evaluate_model(pipeline, X_test, y_test)

    print("\n  -- Classification Report --")
    print(report)
    print("  -- Summary Metrics --")
    for k, v in metrics.items():
        print(f"    {k:>12s}: {v}")

    # Save metrics as JSON
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  [OK] Metrics saved -> {metrics_path}")

    # Save classification report as text
    report_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("PNR Waitlist Confirmation - Classification Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(report + "\n\n")
        f.write("Summary Metrics\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:>12s}: {v}\n")
    print(f"  [OK] Report saved -> {report_path}")

    # ── Phase 5: Save Model ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n  [OK] Trained pipeline saved -> {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  Training pipeline finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
