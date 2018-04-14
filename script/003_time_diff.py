# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import numpy as np
import pandas as pd
from tools import get_group

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat = train.append(test)

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat = concat.merge(concat_category[['instance_id', 'item_category_1']], how='left', on='instance_id')


def last_time_diff(df, cols):
    group = get_group(df, cols)

    last_time = df.groupby(group).context_timestamp.last()

    return np.log1p(group.map(last_time) - df.context_timestamp)


def time_from_prev_expose(df, cols):
    group = get_group(df, cols)

    last_expose = df.groupby(group).context_timestamp.shift(1)

    return np.log1p(df.context_timestamp - last_expose)


def make_feats(begin_date, end_date):

    label = concat[(concat['context_date_day'] < end_date) & (concat['context_date_day'] >= begin_date)].copy()
    label = label.sort_values('context_timestamp').reset_index(drop=True)

    result = pd.DataFrame()

    # result['user_item_last_time_diff'] = last_time_diff(label, ['user_id', 'item_id'])  # 太稀疏
    # result['user_category_last_time_diff'] = last_time_diff(label, ['user_id', 'item_category_1'])

    result['user_from_prev_time_diff'] = time_from_prev_expose(label, ['user_id'])
    result['user_item_from_prev_time_diff'] = time_from_prev_expose(label, ['user_id', 'item_id'])
    # result['shop_id_from_prev_time_diff'] = time_from_prev_expose(label, ['shop_id'])

    #  与user高相关性
    # result['user_category_from_prev_time_diff'] = time_from_prev_expose(label, ['user_id', 'item_category_1'])

    result['instance_id'] = label.instance_id.values
    result = result.set_index('instance_id')

    return result


concat_features = make_feats(18, 26)
concat_features.to_pickle('../features/concat_time_diff_features.p')

