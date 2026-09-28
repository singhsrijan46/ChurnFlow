# Data Management & Storage

This directory manages the datasets used in the Customer Churn Prediction MLOps lifecycle.

## Directory Structure
- `data/raw/`: Contains pristine, untransformed raw datasets (e.g. `telco_churn_raw.csv`). Version-controlled with DVC.
- `data/processed/`: Contains cleaned, split, and transformed datasets (`train.csv`, `test.csv`, `reference.csv`, `current.csv`).

## Dataset Details
- **Dataset**: IBM Telco Customer Churn Dataset
- **Records**: 7,043 customer accounts
- **Target**: `Churn` (Yes = 1, No = 0)
- **Features**: 20 customer demographic, account, and subscribed services attributes
- **Source**: Public IBM Telco Customer Churn open dataset.
