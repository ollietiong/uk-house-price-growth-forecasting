import pandas as pd

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")
    
    return df

def load_data_xls(path:str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=37)

    if df.empty:
        raise ValueError("Loaded dataset is empty")
    
    return df

