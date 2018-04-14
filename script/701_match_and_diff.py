# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import reduce

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

dfAll = pd.concat([train, test])
dfAll['context_shift_hour'] = (dfAll['context_timestamp'] - 1537200001) / (60*60)
dfAll['context_shift_date'] = dfAll['context_shift_hour'].astype(int) // 24
dfAll['context_shift_hour'] = dfAll['context_shift_hour'].astype(int) % 24


def calc_match(dfAll, obj, sub, target):
    """
    Calculate suitability
    :param dfAll:
    :param obj:
    :param sub:
    :param target:
    :return:
    """
    print("Processing {OBJ}-{SUB}-{TARGET} match".format(OBJ=obj, SUB=sub, TARGET=target))
    use_cols = [sub, obj, target]
    data = dfAll.loc[:, use_cols]
    
    sub_tbl = data.groupby(sub).agg({target: np.nanmean}).rename(
        columns={target: "{}_group_{}_mean".format(sub, target)}).reset_index()
    obj_tbl = data.groupby(obj).agg({target: np.nanmean}).rename(
        columns={target: "{}_{}_mean".format(obj, target)}).reset_index()
    
    main_tbl = dfAll.loc[:, ["instance_id", obj, sub, "is_trade"]]
    main_tbl = main_tbl.merge(sub_tbl, how="left").merge(obj_tbl, how="left")
    
    col = '{}_group_{}_{}_diff'.format(sub, obj, target)
    main_tbl[col] = main_tbl["{}_group_{}_mean".format(sub, target)] - main_tbl["{}_{}_mean".format(obj, target)]
    
    return main_tbl.loc[:, ["instance_id", col]]

pairs = [
    ['shop_id', 'user_id', 'item_price_level'],
    ['shop_id', 'user_age_level', 'item_price_level']
]


pool = mp.Pool(30)
output = pool.starmap(calc_match, [(dfAll, o, s, t)  for o,s,t in pairs] )
output = reduce(lambda x, y: pd.merge(x, y, on='instance_id', how='left'), output).set_index("instance_id")
pool.close()

output.to_pickle('../features/match_and_diff.p')
