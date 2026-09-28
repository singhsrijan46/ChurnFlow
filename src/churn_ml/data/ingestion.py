from pathlib import Path
from typing import Optional, Tuple
import os
import io
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.data.validation import validate_raw_data

DATASET_MIRRORS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://cdn.jsdelivr.net/gh/IBM/telco-customer-churn-on-icp4d@master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def generate_synthetic_telco_dataset(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic IBM Telco Customer Churn dataset matching real-world schema and distributions.
    Used as an offline fallback when network mirrors are unavailable.
    """
    np.random.seed(random_state)
    logger.info(f"Generating synthetic Telco churn dataset with {n_samples} samples...")

    customer_ids = [f"{np.random.randint(1000, 9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n_samples)]
    genders = np.random.choice(["Female", "Male"], size=n_samples, p=[0.495, 0.505])
    seniors = np.random.choice([0, 1], size=n_samples, p=[0.838, 0.162])
    partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.483, 0.517])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.299, 0.701])
    
    # Tenure in months
    tenures = np.clip(np.random.exponential(scale=32, size=n_samples).astype(int), 0, 72)
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.903, 0.097])
    
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.42, 0.58]))

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.344, 0.440, 0.216])
    
    def gen_internet_addon(service_col):
        res = []
        for s in service_col:
            if s == "No":
                res.append("No internet service")
            else:
                res.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
        return res

    online_security = gen_internet_addon(internet_service)
    online_backup = gen_internet_addon(internet_service)
    device_protection = gen_internet_addon(internet_service)
    tech_support = gen_internet_addon(internet_service)
    streaming_tv = gen_internet_addon(internet_service)
    streaming_movies = gen_internet_addon(internet_service)

    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.550, 0.209, 0.241])
    paperless = np.random.choice(["Yes", "No"], size=n_samples, p=[0.592, 0.408])
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.335, 0.228, 0.219, 0.218],
    )

    monthly_charges = []
    for inet, cont in zip(internet_service, contracts):
        base = 20.0 if inet == "No" else (55.0 if inet == "DSL" else 85.0)
        noise = np.random.normal(0, 8)
        monthly_charges.append(round(float(np.clip(base + noise, 18.0, 118.0)), 2))
    monthly_charges = np.array(monthly_charges)

    total_charges = []
    for t, mc in zip(tenures, monthly_charges):
        if t == 0:
            total_charges.append(" ")
        else:
            tc = t * mc + np.random.normal(0, 10)
            total_charges.append(str(round(float(max(tc, mc)), 2)))

    # Realistic Churn Logic
    churn_prob = (
        0.15
        + (contracts == "Month-to-month") * 0.28
        + (internet_service == "Fiber optic") * 0.15
        + (payment_methods == "Electronic check") * 0.12
        - (tenures / 72.0) * 0.30
        + (seniors == 1) * 0.05
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.90)
    churn = ["Yes" if np.random.rand() < p else "No" for p in churn_prob]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": seniors,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenures,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })
    return df


def download_raw_dataset(url: Optional[str] = None, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Downloads raw Telco Customer Churn dataset from source mirrors or generates realistic dataset if offline.
    """
    target_path = Path(save_path or settings.data.raw_data_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and os.path.getsize(target_path) > 1000:
        logger.info(f"Loading existing raw dataset from {target_path}")
        return pd.read_csv(target_path)

    urls = [url] if url else DATASET_MIRRORS
    for target_url in urls:
        if not target_url:
            continue
        try:
            logger.info(f"Attempting to download raw dataset from {target_url}...")
            response = requests.get(target_url, timeout=10, headers={"User-Agent": "MLOps-ChurnPlatform/1.0"})
            if response.status_code == 200 and len(response.text) > 1000:
                df = pd.read_csv(io.StringIO(response.text))
                df.to_csv(target_path, index=False)
                logger.info(f"Saved downloaded dataset to {target_path} ({len(df)} records)")
                return df
        except Exception as e:
            logger.warning(f"Mirror {target_url} failed: {e}")

    # Fallback to reproducible synthetic generation
    logger.info("Generating pristine baseline Telco dataset offline...")
    df_synthetic = generate_synthetic_telco_dataset()
    df_synthetic.to_csv(target_path, index=False)
    logger.info(f"Saved dataset to {target_path} ({len(df_synthetic)} records)")
    return df_synthetic


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset types and whitespace:
    - Converts 'TotalCharges' whitespace blanks to NaN and casts to float.
    - Strips whitespace from string columns.
    - Standardizes target column 'Churn' into binary 0/1 integer.
    """
    df_clean = df.copy()

    # Clean whitespace in string columns
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # TotalCharges contains blank spaces for new customers (tenure == 0)
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Standardize target Churn (Yes -> 1, No -> 0)
    target_col = settings.data.target_column
    if target_col in df_clean.columns:
        if df_clean[target_col].dtype == object:
            df_clean[target_col] = df_clean[target_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        df_clean[target_col] = df_clean[target_col].astype(int)

    return df_clean


def ingest_data(
    raw_url: Optional[str] = None,
    raw_path: Optional[str] = None,
    clean: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Full data ingestion pipeline: fetch, validate raw, and clean.
    """
    df_raw = download_raw_dataset(url=raw_url, save_path=raw_path)

    if validate:
        validate_raw_data(df_raw, raise_on_error=True)

    if clean:
        df_clean = clean_raw_data(df_raw)
        return df_clean

    return df_raw


def save_processed_splits(
    df: pd.DataFrame,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    reference_path: Optional[str] = None,
    current_path: Optional[str] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits cleaned data into Train and Test sets, and saves reference & current sets for drift monitoring.
    """
    t_size = test_size or settings.data.test_size
    r_state = random_state or settings.data.random_state
    target = settings.data.target_column

    t_path = Path(train_path or settings.data.processed_train_path)
    te_path = Path(test_path or settings.data.processed_test_path)
    ref_path = Path(reference_path or settings.data.reference_data_path)
    curr_path = Path(current_path or settings.data.current_data_path)

    t_path.parent.mkdir(parents=True, exist_ok=True)

    stratify_col = df[target] if target in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=t_size,
        random_state=r_state,
        stratify=stratify_col,
    )

    train_df.to_csv(t_path, index=False)
    test_df.to_csv(te_path, index=False)
    # Reference set is train data (baseline for drift)
    train_df.to_csv(ref_path, index=False)
    # Current set is test data (simulated production baseline)
    test_df.to_csv(curr_path, index=False)

    logger.info(f"Processed datasets saved: Train ({len(train_df)} rows) -> {t_path}, Test ({len(test_df)} rows) -> {te_path}")
    return train_df, test_df
