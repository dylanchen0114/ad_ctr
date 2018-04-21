# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""

import pandas as pd
from datetime import datetime
import os
import numpy as np

import xgbfir
import xgboost as xgb

train_y = pd.read_pickle('../train_cache/train_y.p')
train_x = pd.read_pickle('../train_cache/train_x.p')

valid_y = pd.read_pickle('../train_cache/valid_y.p')
valid_x = pd.read_pickle('../train_cache/valid_x.p')

feat_cnt = train_x.shape[1]
print('total features: ', feat_cnt)

dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns)
dvalid = xgb.DMatrix(valid_x, valid_y, feature_names=valid_x.columns)

params = {
    'max_depth': 10,
    'eta': 0.05,
    'colsample_bytree': 0.98,
    'subsample': 0.95,
    'min_child_weight': 50,
    'gamma': 0,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'silent': True,
    'seed': 114
}

evals_result = {}
watch_list = [(dtrain, 'train'), (dvalid, 'valid')]
bst = xgb.train(params, dtrain, 10000, watch_list, evals_result=evals_result, verbose_eval=20, early_stopping_rounds=50)

stamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
out_dir = "/home/xiaolan/project/private/contest/alimama_v2/train_log"
xgbfir.saveXgbFI(bst, feature_names=train_x.columns,
                     OutputXlsxFile=os.path.join(out_dir,
                                                 "feature_importance_{}_{:.4f}.xlsx".format(stamp, bst.best_score)))


best_round = np.argmin(evals_result['valid']['logloss'])

val_loss = evals_result['valid']['logloss'][best_round]
trn_loss = evals_result['train']['logloss'][best_round]

record = "{timestamp}, {eta}, {max_depth}, {min_child_weight}, {gamma}, {best_round},{trn_loss},{val_loss}\n".\
format(timestamp=datetime.now().strftime('%m/%d %H:%M'), eta=params['eta'], min_child_weight=params['min_child_weight'], max_depth=params['max_depth'], gamma=params['gamma'], best_round=best_round, val_loss=val_loss, trn_loss=trn_loss)

with open('{}/train_log.csv'.format(out_dir), 'a') as f:
    f.write(record)
