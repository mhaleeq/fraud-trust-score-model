"""
Score a member's fraud/trust risk using the trained model.

Usage:
    python src/predict.py                 # scores a demo example
    python src/predict.py --demo-batch     # scores a few example profiles
"""

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import engineer_features, FEATURE_COLUMNS

MODEL_PATH = "models/trust_score_model.joblib"


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], bundle["name"]


def score_member(member: dict, model, features) -> dict:
    df = pd.DataFrame([member])
    df = engineer_features(df)
    X = df[features]
    risk_proba = model.predict_proba(X)[0, 1]
    trust_score = round((1 - risk_proba) * 100, 1)  # 0 = highest risk, 100 = fully trusted
    band = (
        "high_risk" if risk_proba >= 0.5 else
        "watch" if risk_proba >= 0.25 else
        "trusted"
    )
    return {"risk_probability": round(risk_proba, 4), "trust_score": trust_score, "band": band}


DEMO_PROFILES = [
    {
        "label": "Long-standing consistent contributor",
        "account_age_days": 900, "kyc_completeness": 1.0, "avg_contribution": 60000,
        "contribution_count": 24, "contribution_consistency": 0.95, "num_payment_methods": 2,
        "device_changes_30d": 0, "failed_login_attempts_30d": 0,
        "withdrawal_to_deposit_ratio": 0.1, "referral_count": 3,
        "days_since_last_activity": 5, "velocity_score": 0.15,
    },
    {
        "label": "New account, incomplete KYC, high velocity",
        "account_age_days": 10, "kyc_completeness": 0.5, "avg_contribution": 15000,
        "contribution_count": 1, "contribution_consistency": 0.2, "num_payment_methods": 1,
        "device_changes_30d": 3, "failed_login_attempts_30d": 4,
        "withdrawal_to_deposit_ratio": 1.4, "referral_count": 0,
        "days_since_last_activity": 1, "velocity_score": 0.85,
    },
    {
        "label": "Moderate tenure, some inconsistency",
        "account_age_days": 200, "kyc_completeness": 0.75, "avg_contribution": 30000,
        "contribution_count": 6, "contribution_consistency": 0.55, "num_payment_methods": 2,
        "device_changes_30d": 1, "failed_login_attempts_30d": 1,
        "withdrawal_to_deposit_ratio": 0.4, "referral_count": 1,
        "days_since_last_activity": 40, "velocity_score": 0.4,
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-batch", action="store_true", help="Score example member profiles")
    args = parser.parse_args()

    model, features, name = load_model()
    print(f"Loaded model: {name}\n")

    profiles = DEMO_PROFILES if args.demo_batch else [DEMO_PROFILES[1]]
    for profile in profiles:
        label = profile.pop("label")
        result = score_member(profile, model, features)
        print(f"{label}")
        print(f"  Trust score: {result['trust_score']}/100  |  Risk band: {result['band']}  |  Risk probability: {result['risk_probability']}\n")
