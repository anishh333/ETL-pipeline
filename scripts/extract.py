import os
import pandas as pd

def extract_data():
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir=os.path.join(base_dir,"data","raw")
    os.makedirs(data_dir,exist_ok=True)

    df=pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
    raw_path=os.path.join(data_dir,"Telecom_raw.csv")
    df.to_csv(raw_path,index=False)

    print(f"Data Extracted and saved {raw_path}")
    return raw_path

if __name__=="__main__":
    extract_data()