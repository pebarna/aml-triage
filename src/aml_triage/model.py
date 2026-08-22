import xgboost as xgb

def train_baseline(X_train, Y_train, weight: float) -> xgb.XGBClassifier:
  model = xgb.XGBClassifier(
    n_estimators = 100,
    max_depth = 4,
    learning_rate = 0.1,
    scale_pos_weight = weight,
    eval_metric = "aucpr",
    random_state = 42
  )
  model.fit(X_train, Y_train)
  return model