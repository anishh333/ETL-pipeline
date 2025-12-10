import os
import time
import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
from typing import Any

def get_supabase_client():
    load_dotenv()
    url = os.getenv("supabase_url")
    key = os.getenv("supabase_key")

    if not url or not key:
        raise EnvironmentError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")

    return create_client(url, key)

def _make_json_safe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert numpy types to native Python types and replace NaN/Inf with None
    so the JSON payload is RFC-compliant for PostgREST/Supabase.
    """
    safe_records = []
    for rec in records:
        safe = {}
        for k, v in rec.items():
            # Replace infinities explicitly
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                safe[k] = None
                continue

            # Convert numpy scalar types to native python types
            if isinstance(v, (np.integer, np.int64, np.int32)):
                safe[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                # NaN/Inf handled above; convert float
                safe[k] = float(v)
            elif isinstance(v, (np.bool_ , )):
                safe[k] = bool(v)
            elif isinstance(v, (np.generic,)):
                # generic fallback
                try:
                    safe[k] = v.item()
                except Exception:
                    safe[k] = None
            else:
                safe[k] = v
        safe_records.append(safe)
    return safe_records


def load_to_supabase(staged_path: str, table_name: str = "telecom_transformed", batch_size: int = 200, pause_seconds: float = 0.2):
    # Resolve path relative to this file if needed (keeps your original behavior)
    if not os.path.isabs(staged_path):
        staged_path = os.path.abspath(os.path.join(os.path.dirname(__file__), staged_path))

    print(f"🔍 Looking for data file at: {staged_path}")

    if not os.path.exists(staged_path):
        print(f"❌ Error: File not found at {staged_path}")
        print("ℹ️  Please run transform.py first to generate the transformed data")
        return

    supabase = get_supabase_client()

    df = pd.read_csv(staged_path)

    # Ensure no infinities, then replace NaN with None so JSON uses null
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    totalrows = len(df)
    print(f"ℹ️  Preparing to insert {totalrows} rows into '{table_name}' (batch size: {batch_size})")

    for i in range(0, totalrows, batch_size):
        batch_df = df.iloc[i:i+batch_size].copy()

        # Convert to records and make JSON-safe (native types, no NaN/Inf)
        records = batch_df.to_dict("records")
        records = _make_json_safe(records)

        try:
            response = supabase.table(table_name).insert(records).execute()
        except Exception as e:
            print(f"⚠️  Error in batch {i//batch_size + 1}: {str(e)}")
            # continue with next batch rather than crash
            continue

        # Check response for errors in several possible shapes
        error = None
        status = None
        try:
            status = getattr(response, "status_code", None) or (response.get("status_code") if isinstance(response, dict) else None)
        except Exception:
            status = None

        # supabase-py can return a dict-like object where 'error' key is present
        if isinstance(response, dict):
            error = response.get("error")
        else:
            # try accessing attribute
            error = getattr(response, "error", None)

        if error:
            print(f"⚠️  Error in batch {i//batch_size + 1}: {error}")
        else:
            end = min(i + batch_size, totalrows)
            print(f"✅ Inserted rows {i+1}-{end} of {totalrows}")

        # small pause to avoid rate-limiting when many batches
        time.sleep(pause_seconds)

    print(f"🎯 Finished loading data into '{table_name}'.")

if __name__ == "__main__":
    staged_csv_path = os.path.join("..", "data", "staged", "telecom_transformed.csv")
    load_to_supabase(staged_csv_path)
