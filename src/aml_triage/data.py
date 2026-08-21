import pandas as pd

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

def load_transactions(path: str) -> pd.DataFrame:
  df = pd.read_csv(path)
  missing = set(EXPECTED_COLUMNS) - set(df.columns)
  if missing:
    raise ValueError(f"missing expected columns: {missing}")
  # coerce dtypes
  df["step"] = df["step"].astype("int64")
  df["amount"] = df["amount"].astype("float64")
  df["oldbalanceOrg"] = df["oldbalanceOrg"].astype("float64")
  df["newbalanceOrig"] = df["newbalanceOrig"].astype("float64")
  df["oldbalanceDest"] = df["oldbalanceDest"].astype("float64")
  df["newbalanceDest"] = df["newbalanceDest"].astype("float64")
  df["isFraud"] = df["isFraud"].astype("int16")
  df["isFlaggedFraud"] = df["isFlaggedFraud"].astype("int16")
  return df