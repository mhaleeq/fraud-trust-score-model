# Member Trust / Fraud Risk Scoring Model

A machine learning model that scores members of a cooperative investment platform
(e.g. a fund manager / investment club like Vizient Coop) on a 0–100 trust scale,
based on account behavior, KYC completeness, contribution patterns, and activity signals.

> **Note:** This project uses fully **synthetic data**, generated to mimic realistic
> patterns in a contribution-based fintech/asset management platform. No real member
> or financial data is used. Built as a portfolio project to demonstrate applied risk
> modeling relevant to fund management and financial services.

## Problem

Asset managers and investment cooperatives need to flag potentially risky or
fraudulent member behavior early, without adding friction for the vast majority
of legitimate, trustworthy members. This project frames that as a binary
classification problem: predict the probability a member profile represents
elevated risk, then convert that into an interpretable trust score.

## Approach

1. **Synthetic data generation** (`data/generate_data.py`) — simulates 5,000 member
   profiles with account age, KYC completeness, contribution consistency, payment
   method diversity, device/login anomalies, withdrawal behavior, and activity recency.
   Risk labels are generated from a weighted combination of realistic red flags
   (incomplete KYC, brand-new accounts, high fund velocity, login/device anomalies,
   high withdrawal-to-deposit ratios, low contribution consistency) plus noise, so
   the signal is realistic rather than trivially separable.

2. **Feature engineering** (`src/features.py`) — adds derived flags such as
   `is_new_account` and `activity_gap_flag`.

3. **Model training** (`src/train.py`) — trains and compares two models:
   - Logistic Regression (scaled, class-balanced) — interpretable baseline
   - Random Forest (class-balanced) — captures non-linear interactions

   Selects the best model by ROC-AUC on a held-out test set, saves it, and
   generates evaluation plots (confusion matrix, ROC curve, feature importance).

4. **Scoring** (`src/predict.py`) — loads the saved model and converts a member's
   predicted risk probability into a 0–100 trust score with a risk band
   (`trusted` / `watch` / `high_risk`).

## Results

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** (selected) | **0.715** |
| Random Forest | 0.657 |

Logistic Regression was selected both for stronger ROC-AUC and because its
coefficients are directly explainable to non-technical stakeholders — an
important property for a financial services context where risk decisions need
to be justified.

**Top risk drivers** (by feature importance):
1. Velocity score (speed of funds moving in/out)
2. KYC completeness
3. Withdrawal-to-deposit ratio
4. Device changes in the last 30 days
5. Failed login attempts in the last 30 days

Full results in `outputs/evaluation_plots.png` and `outputs/feature_importance.png`.

### Example scores

| Profile | Trust Score | Band |
|---|---|---|
| Long-standing, consistent contributor | 86.4 / 100 | trusted |
| Moderate tenure, some inconsistency | 31.2 / 100 | high_risk |
| New account, incomplete KYC, high velocity | 0.2 / 100 | high_risk |

## How to run

```bash
pip install -r requirements.txt

# 1. Generate synthetic data
python data/generate_data.py

# 2. Train and evaluate the model
python src/train.py

# 3. Score example member profiles
python src/predict.py --demo-batch
```

## Project structure

```
fraud-trust-score-model/
├── data/
│   └── generate_data.py       # synthetic data generator
├── src/
│   ├── features.py            # feature engineering
│   ├── train.py                # model training & evaluation
│   └── predict.py              # scoring / inference
├── models/                     # saved model artifact (generated)
├── outputs/                    # evaluation plots & metrics (generated)
├── requirements.txt
└── README.md
```

## Possible extensions

- Swap in real transaction-level data and calibrate the risk labels with actual
  fraud/chargeback outcomes.
- Add SHAP values for per-prediction explainability.
- Serve the model behind a lightweight API (FastAPI) for real-time scoring.
- Add a monitoring layer to track score drift as member behavior evolves.

## Author

Abdulmalik Wahab — Data Science  | Machine Learning and AI | Pharmacology | Health & fintech
analytics
