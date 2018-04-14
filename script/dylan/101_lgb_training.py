# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

train_y = pd.read_hdf('../for_fit/full_train_y.hdf')
train_x = pd.read_hdf('../for_fit/full_train_x.hdf')

test_x = pd.read_hdf('../for_fit/test_x.hdf')
test_id = pd.read_hdf('../for_fit/test_id.hdf')

print('check_test_id_seq', sum(test_x.instance_id.values == test_id.values))

test_x.drop('instance_id', axis=1, inplace=True)

train_data = lgb.Dataset(train_x, label=train_y)

embedding_features = ['user_gender_id', 'user_occupation_id', 'user_age_level']

for col in embedding_features:
    train_x[col] = train_x[col].astype('category')
    test_x[col] = test_x[col].astype('category')

print('total full_train: ', train_x.shape, '\n')
print('total test: ', test_x.shape, '\n')


params = {

    'boosting_type': 'gbdt',  # np.random.choice(['dart', 'gbdt']),
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'max_bin': 256,

    'learning_rate': 0.02,

    'num_leaves': 100,
    'max_depth': 12,
    'min_data_in_leaf': 1000,

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

num_rounds = 369

gbm = lgb.train(params, train_data, num_boost_round=num_rounds, valid_sets=[train_data],
                early_stopping_rounds=250, verbose_eval=50)

pred_train = gbm.predict(train_x)
trn_loss = log_loss(y_pred=pred_train, y_true=train_y)
trn_auc = roc_auc_score(y_score=pred_train, y_true=train_y)

print('Training loss: %.5f, Training AUC: %.5f' % (trn_loss, trn_auc))

feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()}).sort_values(
    by='importance', ascending=False)
feature_importance.to_csv('../full_train_feat_importance_with_leak.csv', index=False)

pred_test = gbm.predict(test_x)
test_sub = pd.DataFrame({'instance_id': test_id, 'predicted_score': pred_test})

test_sub.to_csv('../lgb_%.5f_logloss_leak.txt' % trn_loss, index=False, sep=' ', line_terminator='\n')
