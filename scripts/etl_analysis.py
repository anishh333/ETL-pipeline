#!/usr/bin/env python3
"""
etl_analysis.py

Fetch data from Supabase (table: telecom_transformed), compute analytics,
and save a summary CSV with key metrics.

Output:
  - data/processed/analysis_summary.csv
  - data/processed/pivot_churn_vs_tenure_group.csv
"""

import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# ------------ Config ------------
TABLE_NAME = "telecom_transformed"
OUTPUT_DIR = os.path.join("data", "processed")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "analysis_summary.csv")
PIVOT_FILE = os.path.join(OUTPUT_DIR, "pivot_churn_vs_tenure_group.csv")
# --------------------------------

def get_supabase_client():
    load_dotenv()
    url = os.getenv("supabase_url")
    key = os.getenv("supabase_key")
    if not url or not key:
        raise EnvironmentError("Missing SUPABASE_URL or SUPABASE_KEY.")
    return create_client(url, key)

def fetch_table_as_df(supabase, table_name):
    """Read entire table into a DataFrame."""
    resp = supabase.table(table_name).select("*").execute()

    # Handle dict or object response
    if isinstance(resp, dict):
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        data = resp.get("data")
    else:
        if getattr(resp, "error", None):
            raise RuntimeError(resp.error)
        data = getattr(resp, "data", None) or getattr(resp, "body", None)

    if data is None:
        raise RuntimeError("No data returned from Supabase.")

    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def normalize_churn(df):
    """
    Normalize Churn column into churn_flag (0/1).
    Accepts Yes/No, 1/0, True/False.
    """
    churn_col_candidates = [c for c in df.columns if c.lower() == "churn"]
    if not churn_col_candidates:
        raise KeyError("Churn column not found.")
    churn_col = churn_col_candidates[0]

    def to_binary(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ("yes", "y", "true", "1"):
            return 1
        if s in ("no", "n", "false", "0"):
            return 0
        try:
            f = float(s)
            return 1 if f == 1 else 0 if f == 0 else np.nan
        except:
            return np.nan

    df["churn_flag"] = df[churn_col].map(to_binary).astype("Float64")
    return df

def compute_metrics(df):
    """Compute required ETL analytics."""

    df = normalize_churn(df)

    # Helpful alias map
    cols = {c.lower(): c for c in df.columns}

    monthly = cols.get("monthlycharges")
    total = cols.get("totalcharges")
    tenure_group = cols.get("tenure_group")
    monthly_segment = cols.get("monthly_charge_segment")
    internet = cols.get("has_internet_service")
    contract = cols.get("contract_months")

    if monthly:
        df[monthly] = pd.to_numeric(df[monthly], errors='coerce')

    # ---- 1. Churn percentage ----
    churn_count = df["churn_flag"].dropna().sum()
    churn_total = df["churn_flag"].notna().sum()
    churn_pct = float(churn_count) / churn_total * 100 if churn_total > 0 else np.nan

    # ---- 2. Average monthly charges per contract type ----
    avg_monthly_by_contract = {}
    if contract and monthly:
        grp = df.groupby(contract)[monthly].mean()
        avg_monthly_by_contract = grp.fillna(np.nan).to_dict()

    # ---- 3. Counts by tenure group ----
    counts_by_tenure = {}
    if tenure_group:
        counts = df[tenure_group].astype(str).replace("nan", np.nan)
        counts_by_tenure = counts.value_counts(dropna=True).to_dict()

    # ---- 4. Internet service distribution ----
    internet_dist = {}
    if internet:
        dist = df[internet].value_counts(dropna=True)
        internet_dist = {str(k): int(v) for k, v in dist.items()}

    # ---- 5. Pivot: Churn vs Tenure Group ----
    pivot_df = None
    if tenure_group:
        pc = pd.crosstab(df[tenure_group], df["churn_flag"], dropna=False)
        pc = pc.reindex(columns=[0.0, 1.0], fill_value=0)
        pc["churn_rate_pct"] = (pc[1.0] / (pc[0.0] + pc[1.0]).replace({0: np.nan})) * 100
        pivot_df = pc.reset_index()

    # ---- Prepare summary dataframe ----
    summary_row = {
        "total_records": len(df),
        "records_with_churn": int(churn_total),
        "churn_percentage": churn_pct,
        "avg_monthly_by_contract": json.dumps(avg_monthly_by_contract),
        "counts_by_tenure_group": json.dumps(counts_by_tenure),
        "internet_service_distribution": json.dumps(internet_dist),
    }

    summary_df = pd.DataFrame([summary_row])
    return summary_df, pivot_df

def main():
    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    supabase = get_supabase_client()
    print(f"🔍 Fetching '{TABLE_NAME}' from Supabase...")

    df = fetch_table_as_df(supabase, TABLE_NAME)
    if df.empty:
        print("⚠️ Table is empty. Nothing to analyze.")
        return

    summary_df, pivot_df = compute_metrics(df)

    # Save summary file
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"✅ Summary saved: {SUMMARY_FILE}")

    # Save pivot
    if pivot_df is not None:
        pivot_df.to_csv(PIVOT_FILE, index=False)
        print(f"✅ Pivot saved: {PIVOT_FILE}")

    print("🎯 Analysis completed successfully.")
    print(summary_df)

if __name__ == "__main__":
    main()
