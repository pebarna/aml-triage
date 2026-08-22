import pandas as pd

def compute_scale_pos_weight(y: pd.Series) -> float:
  positive = (y == 1).sum()
  negative = (y == 0).sum()
  if positive == 0:
    raise ValueError("cannot compute scale_pos_weight: no positive examples in y")
  return negative / positive
  