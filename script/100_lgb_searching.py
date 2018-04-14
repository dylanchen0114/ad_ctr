# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

train_y = pd.read_hdf('../for_fit/train_y.hdf')
train_x = pd.read_hdf('../for_fit/train_x.hdf')

valid_y = pd.read_hdf('../for_fit/valid_y.hdf')
valid_x = pd.read_hdf('../for_fit/valid_x.hdf')

train_data = lgb.Dataset(train_x, label=train_y)
valid_data = lgb.Dataset(valid_x, label=valid_y, reference=train_data)

feat_cnt = train_x.shape[1]
print('total features: ', feat_cnt)

embedding_features = ['user_gender_id']

for col in embedding_features:
    train_x[col] = train_x[col].astype('category')
    valid_x[col] = valid_x[col].astype('category')

num_rounds = 5000
params = {

    'boosting_type': 'gbdt',  # np.random.choice(['dart', 'gbdt']),
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'max_bin': 256,

    'learning_rate': 0.02,

    'num_leaves': 100,
    'max_depth': 12,
    'min_data_in_leaf': 200,

    'feature_fraction': 0.7,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,

    'lambda_l1': 0,
    'lambda_l2': 0,
    'min_gain_to_split': 0.0,
    'min_sum_hessian_in_leaf': 0.1,

    'verbose': 1,
    'is_training_metric': 'True'
}


evals_result = {}

gbm = lgb.train(params, train_data, num_boost_round=num_rounds, valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'], evals_result=evals_result,
                early_stopping_rounds=150, verbose_eval=50)

bst_round = np.argmin(evals_result['valid']['binary_logloss'])

trn_loss = evals_result['train']['binary_logloss'][bst_round]
val_loss = evals_result['valid']['binary_logloss'][bst_round]

print('Best Round: %d' % bst_round)
print('Training loss: %.5f, Validation loss: %.5f' % (trn_loss, val_loss))


feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()}).sort_values(
    by='importance', ascending=False)
feature_importance.to_csv('../feat_importance_with_leak.csv', index=False)

res = '%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%s,%.5f,%.5f\n' % \
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'with_three_prev_time_diff', feat_cnt, params['boosting_type'],
           params['learning_rate'], params['num_leaves'], params['max_depth'],
           params['min_data_in_leaf'], params['feature_fraction'], params['bagging_fraction'],
           params['bagging_freq'], params['lambda_l1'], params['lambda_l2'], params['min_gain_to_split'],
           params['min_sum_hessian_in_leaf'], 0.0, bst_round+1, trn_loss, val_loss)

with open('../lgb_record.csv', 'a') as f:
    f.write(res)
