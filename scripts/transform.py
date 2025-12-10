import os
import pandas as pd
import numpy as np

def transform_data(raw_path, staged_filename="telecom_transformed.csv"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "staged")
    os.makedirs(data_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    # normalize column names to lowercase (helps avoid DB/rest casing issues)
    df.columns = [c.lower() for c in df.columns]

    # TotalCharges: convert blanks to NaN, coerce to numeric, fill missing with median
    if 'totalcharges' in df.columns:
        df['totalcharges'] = df['totalcharges'].replace(' ', np.nan)
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].median())

    # -- Data Transformations
    # tenure grouping (assuming 'tenure' column exists)
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 36, 60, float('inf')],
            labels=['New', 'Regular', 'Loyal', 'Champion'],
            right=True,
            include_lowest=True
        )

    # monthly charge segment
    if 'monthlycharges' in df.columns:
        df['monthly_charge_segment'] = pd.cut(
            df['monthlycharges'],
            bins=[0, 30, 70, float('inf')],
            labels=['Low', 'Medium', 'High'],
            right=True,
            include_lowest=True
        )

    # Internet service: DSL or Fiber Optic -> 1 else 0
    if 'internetservice' in df.columns:
        df['has_internet_service'] = df['internetservice'].apply(
            lambda t: 1 if str(t).strip().lower() in {'dsl', 'fiber optic', 'fiber', 'fiberoptic'} else 0
        )

    # Multiple lines: Yes -> 1 else 0
    if 'multiplelines' in df.columns:
        df['is_multiline'] = df['multiplelines'].apply(lambda t: 1 if str(t).strip().lower() == 'yes' else 0)

    # Contract mapping: Month-to-month -> 0, One year -> 1, Two year -> 2
    if 'contract' in df.columns:
        mapping = {
            'month-to-month': 0,
            'one year': 1,
            'two year': 2
        }
        # normalize strings, map, fallback to 0 (or None depending on desired behavior)
        df['contract_months'] = df['contract'].astype(str).str.strip().str.lower().map(mapping)
        # if any unmapped -> set to None (JSON-safe) or 0 as default:
        df['contract_months'] = df['contract_months'].where(pd.notnull(df['contract_months']), None)

    # -- Drop irrelevant columns (ignore if not present)
    drop_cols = [
        'customerid','gender','seniorcitizen','partner','dependents','phoneservice',
        'onlinesecurity','onlinebackup','deviceprotection','techsupport',
        'streamingtv','streamingmovies','paperlessbilling','multiplelines'
    ]
    df.drop(columns=drop_cols, axis=1, inplace=True, errors='ignore')

    # -- Remove infinities and convert NaN -> None for JSON compliance
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    staged_path = os.path.join(data_dir, staged_filename)
    df.to_csv(staged_path, index=False)
    print(f"✅ Data transformed and saved at: {staged_path}")
    return staged_path

if __name__ == "__main__":
    from extract import extract_data
    rawpath = extract_data()
    transform_data(rawpath, staged_filename="telecom_transformed.csv")