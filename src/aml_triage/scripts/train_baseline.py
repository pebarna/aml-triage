import json

from aml_triage.data import load_transactions
from aml_triage.split import temporal_split
from aml_triage.features import add_features
from aml_triage.imbalance import compute_scale_pos_weight
from aml_triage.model import train_baseline
from aml_triage.evaluate import report


df = load_transactions('data/paysim_sample.csv')
train_df, test_df = temporal_split(df, split_step=355)
train_df = add_features(train_df)
test_df = add_features(test_df)

X_train = train_df.drop(columns=['isFraud'])
y_train = train_df['isFraud']

weight = compute_scale_pos_weight(y_train)
print('scale_pos_weight:', weight)

model = train_baseline(X_train, y_train, weight)
print('trained:', model)

X_test = test_df.drop(columns=["isFraud"])
Y_test = test_df["isFraud"]

scores = model.predict_proba(X_test)[:, 1]
result = report(Y_test, scores, {"min_precision": 0.90})
print(result)

json.dump(result, open("reports/phase1_report.json", "w"), indent=2)