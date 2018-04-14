# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""
import pandas as pd
from functools import reduce

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

drop_columns = ['item_id', 'item_brand_id', 'item_city_id',
                'user_id', 'context_id', 'context_timestamp',
                'shop_id', 'is_trade', 'context_date',
                'context_date_day', 'context_date_hour']

concat = train.append(test)

feature_list = ['concat_time_diff_features.p',
                'oneshot_proba.p',
                'match_and_diff.p']

concat_features = [pd.read_pickle('../features/{}'.format(p)).reset_index() for p in feature_list]

concat = reduce(lambda x, y: x.merge(y, how='left', on='instance_id'), concat_features,concat)

mask = (concat.context_date_day == 24)
valid_y = concat.loc[mask, 'is_trade'].reset_index(drop=True).copy()
valid_x = concat.loc[mask].drop(['instance_id'] + drop_columns, axis=1).reset_index(drop=True).copy()

mask = (concat.context_date_day > 18) & (concat.context_date_day < 24)
train_y = concat.loc[mask, 'is_trade'].reset_index(drop=True).copy()
train_x = concat.loc[mask].drop(['instance_id'] + drop_columns, axis=1).reset_index(drop=True).copy()

mask = (concat.context_date_day == 25)
test_x = concat.loc[mask].drop(drop_columns, axis=1).reset_index(drop=True).copy()

mask = (concat.context_date_day > 18) & (concat.context_date_day < 25)
full_train_y = concat.loc[mask, 'is_trade'].reset_index(drop=True).copy()
full_train_x = concat.loc[mask].drop(['instance_id'] + drop_columns, axis=1).reset_index(drop=True).copy()

print('valid', valid_x.shape)
print('train', train_x.shape)
print('full_train', full_train_x.shape)

print('test', test_x.shape, 'with_instance_id')

print(list(train_x))

train_y.to_pickle('../train_cache/train_y.p')
train_x.to_pickle('../train_cache/train_x.p')

valid_y.to_pickle('../train_cache/valid_y.p')
valid_x.to_pickle('../train_cache/valid_x.p')

full_train_y.to_pickle('../train_cache/full_train_y.p')
full_train_x.to_pickle('../train_cache/full_train_x.p')

test_id = test['instance_id']
test_id.to_pickle('../train_cache/test_id.p')
test_x.to_pickle('../train_cache/test_x.p')
