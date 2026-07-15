"""
Train and evaluate a member trust/fraud risk scoring model.

Compares Logistic Regression (interpretable baseline) against Random Forest
(captures non-linear interactions), selects the better model by ROC-AUC,
and saves it for scoring new members via predict.py.
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import get_feature_matrix, FEATURE_COLUMNS

DATA_PATH = "data/member_data.csv"
MODEL_PATH = "models/trust_score_model.joblib"
OUTPUTS_DIR = "outputs"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X, y = get_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = {"model": model, "auc": auc, "proba": proba}
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {auc:.4f}")
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, target_names=["low_risk", "high_risk"]))

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    print(f"\nSelected best model: {best_name} (ROC-AUC {results[best_name]['auc']:.4f})")

    joblib.dump({"model": best_model, "features": FEATURE_COLUMNS, "name": best_name}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix ({best_name})")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticks([0, 1], ["low_risk", "high_risk"])
    axes[0].set_yticks([0, 1], ["low_risk", "high_risk"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    RocCurveDisplay.from_predictions(y_test, results[best_name]["proba"], ax=axes[1])
    axes[1].set_title("ROC Curve")

    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/evaluation_plots.png", dpi=150)
    print(f"Saved evaluation plots to {OUTPUTS_DIR}/evaluation_plots.png")

    # ---- Feature importance ----
    if best_name == "random_forest":
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.named_steps["clf"].coef_[0])

    imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"], color="#2b6cb0")
    plt.title(f"Feature Importance ({best_name})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/feature_importance.png", dpi=150)
    print(f"Saved feature importance plot to {OUTPUTS_DIR}/feature_importance.png")

    imp_df.sort_values("importance", ascending=False).to_csv(
        f"{OUTPUTS_DIR}/feature_importance.csv", index=False
    )


if __name__ == "__main__":
    main()
