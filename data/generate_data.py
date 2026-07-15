"""
Synthetic data generator for a member trust / fraud risk scoring model.

Simulates member and transaction data for a cooperative investment platform
(e.g. contribution-based fund management, similar to Vizient Coop / Volition Cap).
Fraud/risk patterns are injected deliberately so the model has real signal to learn.

This is SYNTHETIC data generated for portfolio/demo purposes only.
No real member or financial data is used.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate_members(n_members: int = 5000) -> pd.DataFrame:
    member_id = np.arange(1, n_members + 1)

    account_age_days = RNG.integers(1, 2000, n_members)
    kyc_completeness = RNG.choice([0.5, 0.75, 1.0], n_members, p=[0.1, 0.25, 0.65])
    avg_contribution = np.round(RNG.gamma(shape=2.0, scale=25000, size=n_members), 2)
    contribution_count = RNG.poisson(lam=8, size=n_members) + 1
    contribution_consistency = np.clip(RNG.normal(0.75, 0.2, n_members), 0, 1)  # regularity of contribution timing
    num_payment_methods = RNG.integers(1, 4, n_members)
    device_changes_30d = RNG.poisson(lam=0.5, size=n_members)
    failed_login_attempts_30d = RNG.poisson(lam=0.3, size=n_members)
    withdrawal_to_deposit_ratio = np.clip(RNG.normal(0.3, 0.25, n_members), 0, 3)
    referral_count = RNG.poisson(lam=1.2, size=n_members)
    days_since_last_activity = RNG.integers(0, 400, n_members)
    velocity_score = np.clip(RNG.normal(0.4, 0.25, n_members), 0, 1)  # speed of fund movement in/out

    df = pd.DataFrame({
        "member_id": member_id,
        "account_age_days": account_age_days,
        "kyc_completeness": kyc_completeness,
        "avg_contribution": avg_contribution,
        "contribution_count": contribution_count,
        "contribution_consistency": np.round(contribution_consistency, 3),
        "num_payment_methods": num_payment_methods,
        "device_changes_30d": device_changes_30d,
        "failed_login_attempts_30d": failed_login_attempts_30d,
        "withdrawal_to_deposit_ratio": np.round(withdrawal_to_deposit_ratio, 3),
        "referral_count": referral_count,
        "days_since_last_activity": days_since_last_activity,
        "velocity_score": np.round(velocity_score, 3),
    })

    # ---- Inject latent fraud/risk signal ----
    # Higher risk score is driven by a realistic combination of red flags:
    # incomplete KYC, new accounts, high velocity, many device/login anomalies,
    # high withdrawal ratio, low consistency.
    risk_logit = (
        -3.0
        + 2.5 * (1 - df["kyc_completeness"])
        + 1.8 * (df["account_age_days"] < 30).astype(int)
        + 2.0 * df["velocity_score"]
        + 0.35 * df["device_changes_30d"]
        + 0.45 * df["failed_login_attempts_30d"]
        + 1.5 * df["withdrawal_to_deposit_ratio"]
        - 1.2 * df["contribution_consistency"]
        - 0.10 * df["num_payment_methods"]
        + RNG.normal(0, 0.6, n_members)  # noise
    )
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    df["is_high_risk"] = (RNG.random(n_members) < risk_prob).astype(int)

    return df


if __name__ == "__main__":
    df = generate_members(5000)
    df.to_csv("data/member_data.csv", index=False)
    print(f"Generated {len(df)} synthetic member records")
    print(f"High-risk rate: {df['is_high_risk'].mean():.2%}")
    print(df.head())
