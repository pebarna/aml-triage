import pandas as pd

def temporal_split(df: pd.DataFrame, split_step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
  train_df = df[df["step"] <= split_step].reset_index(drop=True)
  test_df = df[df["step"] > split_step].reset_index(drop=True)
  return train_df, test_df