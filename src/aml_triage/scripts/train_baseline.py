from aml_triage.data import load_transactions
from aml_triage.split import temporal_split
from aml_triage.features import add_features
from aml_triage.imbalance import compute_scale_pos_weight
from aml_triage.model import train_baseline

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