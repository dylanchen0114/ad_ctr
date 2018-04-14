# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

dfAll = pd.concat([train, test])
dfAll['context_shift_hour'] = (dfAll['context_timestamp'] - 1537200001) / (60*60)
dfAll['context_shift_date'] = dfAll['context_shift_hour'].astype(int) // 24
dfAll['context_shift_hour'] = dfAll['context_shift_hour'].astype(int) % 24


def calc_oneshot(data, key, end_date=7, start_date=0):
    """
    看一次就下单的人 占 所有下单的人数的比例；曝光一次就下单的概率
    :param data:
    :param end_date: not including the end date
    :param start_date: including the start date
    :return:
    """
    
    history_df = data[(data.context_shift_date < end_date) & (data.context_shift_date >= start_date)].copy()
    
    t = history_df.groupby([key, 'user_id']).agg({'is_trade': ['size', sum]}).reset_index()
    t.columns = [key, 'user_id', 'view_pv', 'cart_pv']
    
    one_shot_uv = t.loc[(t.view_pv == t.cart_pv) & (t.cart_pv >= 1)].groupby([key])['user_id'].apply(
        lambda x: len(x)).rename('oneshot_cart_uv')
    
    ttl_cart_uv = t.loc[t.cart_pv >= 1].groupby([key])['user_id'].size().rename('ttl_cart_uv')
    one_shot_ratio = pd.concat([ttl_cart_uv, one_shot_uv], axis=1).fillna(0)
    one_shot_ratio['{}_oneshot_ratio'.format(key)] = one_shot_ratio['oneshot_cart_uv'] / one_shot_ratio['ttl_cart_uv']
    
    one_shot_proba = t.loc[(t.view_pv == 1)].groupby(key)['cart_pv'].mean().rename('{}_oneshot_proba'.format(key))
    output = pd.concat([one_shot_ratio, one_shot_proba], axis=1).fillna(0)
    
    # return output.loc[:, ['{}_oneshot_ratio'.format(key), '{}_oneshot_proba'.format(key)]].reset_index()
    return output



def cumulative_proba(key):
    """
    cumulative one-shot
    :return:
    """
    output = []
    time_range = [(0, t) for t in range(1, 8)]
    for (start, end) in tqdm(time_range):
        df = calc_oneshot(dfAll, key, end, start)
        df['context_shift_date'] = end
        output.append(df)
    return output

key = 'item_id'
output = cumulative_proba(key=key)
output = pd.concat(output).drop(['ttl_cart_uv', 'oneshot_cart_uv'], axis=1).rename_axis(key).reset_index()
output = dfAll.loc[:, ['instance_id', key, 'context_shift_date']].merge(output, how='left')
output = output.drop([key, 'context_shift_date'], axis=1)

dfAll.loc[:,['instance_id']].merge(output,how='left').set_index('instance_id').to_pickle(
    '../features/oneshot_proba.p')
