# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd


train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

drop_columns = ['item_id', 'item_brand_id', 'item_city_id',
                'user_id', 'context_id', 'context_timestamp',
                'shop_id', 'is_trade', 'context_date',
                'context_date_day', 'context_date_hour']

concat = train.append(test)


mask = (train.context_date_day == 24)
valid_y = train.loc[mask, 'is_trade'].reset_index(drop=True).copy()
valid_x = train.loc[mask].drop(['instance_id'] + drop_columns, axis=1).\
    reset_index(drop=True).copy()

mask = (train.context_date_day > 18) & (train.context_date_day < 24)
train_y = train.loc[mask, 'is_trade'].reset_index(drop=True).copy()
train_x = train.loc[mask].drop(['instance_id'] + drop_columns, axis=1).\
    reset_index(drop=True).copy()

mask = (concat.context_date_day == 25)
test_x = concat.loc[mask].drop(drop_columns, axis=1).\
    reset_index(drop=True).copy()

mask = (concat.context_date_day > 18) & (concat.context_date_day < 25)
full_train_y = concat.loc[mask, 'is_trade'].reset_index(drop=True).copy()
full_train_x = concat.loc[mask].drop(['instance_id'] + drop_columns, axis=1).reset_index(drop=True).copy()


print('valid', valid_x.shape)
print('train', train_x.shape)
print('full_train', full_train_x.shape)

print('test', test_x.shape, 'with_instance_id')

print(list(train_x))


train_y.to_hdf('../for_fit/train_y.hdf', key='abc')
train_x.to_hdf('../for_fit/train_x.hdf', key='abc')

valid_y.to_hdf('../for_fit/valid_y.hdf', key='abc')
valid_x.to_hdf('../for_fit/valid_x.hdf', key='abc')

full_train_y.to_hdf('../for_fit/full_train_y.hdf', key='abc')
full_train_x.to_hdf('../for_fit/full_train_x.hdf', key='abc')

test_id = test['instance_id']
test_id.to_hdf('../for_fit/test_id.hdf', key='abc')
test_x.to_hdf('../for_fit/test_x.hdf', key='abc')
