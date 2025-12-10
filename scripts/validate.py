import pandas as pd
import numpy as np
from supabase import Client
import os

def validate_data(supabase: Client, table_name: str='telecom_transformed'):
    data = supabase.table(table_name).select('*').execute()
    
    df = pd.DataFrame(data.data)
    
    # missing values in:
    # tenure, MonthlyCharges, TotalCharges
    missing_values_tenure = df['tenure'].isna().sum()
    missing_values_in_monthly_charges = df['monthlycharges'].isna().sum()
    missing_total_charges = df['totalcharges'].isna().sum()
    
    # Unique count of rows
    unique_rows = len(df) - df.duplicated().sum()
    
    # Values in contract codes
    contract_code_check = df['contract_months'].isin([0, 1, 2]).all()

    
    # Column checks
    tenure_group_check = 'tenure_group' in df.columns
    monthly_charge_segment_check = 'monthly_charge_segment' in df.columns
    
    print("Data Validation Summary")
    print("=" * 40)
    print(f"Missing values in tenure: {missing_values_tenure}")
    print(f"Missing values in monthly charges: {missing_values_in_monthly_charges}")
    print(f"Missing values in total charges: {missing_total_charges}")
    print(f"Unique rows: {unique_rows}")
    print(f"Contract code valid values (0, 1, 2): {contract_code_check}")
    print(f"tenure_group column exists: {tenure_group_check}")
    print(f"monthly_charge_segment column exists: {monthly_charge_segment_check}")
    
if __name__ == "__main__":
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
    client = create_client(os.getenv('supabase_url'), os.getenv('supabase_key'))
    validate_data(client)