"""Feature engineering for the trust/fraud scoring model."""

import pandas as pd


FEATURE_COLUMNS = [
    "account_age_days",
    "kyc_completeness",
    "avg_contribution",
    "contribution_count",
    "contribution_consistency",
    "num_payment_methods",
    "device_changes_30d",
    "failed_login_attempts_30d",
    "withdrawal_to_deposit_ratio",
    "referral_count",
    "days_since_last_activity",
    "velocity_score",
    "is_new_account",
    "activity_gap_flag",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that help the model separate risk levels."""
    df = df.copy()
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    df["activity_gap_flag"] = (df["days_since_last_activity"] > 180).astype(int)
    return df


def get_feature_matrix(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[FEATURE_COLUMNS]
    y = df["is_high_risk"] if "is_high_risk" in df.columns else None
    return X, y
