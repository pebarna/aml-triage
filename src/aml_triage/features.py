import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.drop(columns = ["nameOrig", "nameDest", "isFlaggedFraud"])
  df["orig_balance_delta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
  df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
  df["is_transfer"] = (df["type"] == "TRANSFER").astype("int64")
  df["is_cash_out"] = (df["type"] == "CASH_OUT").astype("int64")
  df = df.drop(columns=["type"])
  return df