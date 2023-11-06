import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

X_train = pd.read_csv('./data/train.csv')
X_train = X_train.drop_duplicates()
X_train['TotalSpent'] = pd.to_numeric(X_train['TotalSpent'], errors='coerce')
X_train['TotalSpent'] = X_train['TotalSpent'].fillna(X_train['TotalSpent'].mean())

Y_train = X_train.Churn
del X_train['Churn']

dv = DictVectorizer(sparse=False)

dicts = X_train.to_dict(orient='records')
df_tr = dv.fit_transform(dicts)


features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(df_tr, label=Y_train)

xgb_params = {
    'eta':0.006,
    'max_depth': 5,
    'min_child_weight': 6,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=250,
                  verbose_eval=5)

y_pred = model.predict(dtrain)
auc = roc_auc_score(Y_train, y_pred)
print(f'auc={auc}')

# save model

import pickle
output_file = 'model_tree.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)