# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""
import multiprocessing as mp

import pandas as pd

from utils import BayesianSmoothing

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

dfAll = pd.concat([train, test])
dfAll['context_shift_hour'] = (dfAll['context_timestamp'] - 1537200001) / (60*60)
dfAll['context_shift_date'] = dfAll['context_shift_hour'].astype(int) // 24
dfAll['context_shift_hour'] = dfAll['context_shift_hour'].astype(int) % 24


def smoothed_conversion(dfAll, col):
    """
    For high cardinality feature, calculate smoothed expected conversion rate and shift by one day
    :param dfAll:
    :param col:
    :return:
    """
    
    use_cols = ['context_shift_date', col, 'is_trade']
    impression_tbl = dfAll.loc[:, use_cols].copy()
    click_tbl = dfAll.loc[dfAll.is_trade == 1, use_cols].copy()
    
    impression_tbl = pd.crosstab(impression_tbl[col], impression_tbl['context_shift_date'])
    
    impression_tbl_dict = {}
    for item, record in impression_tbl.iterrows():
        impression_tbl_dict[item] = record.tolist()
    
    click_tbl = pd.crosstab(click_tbl[col], click_tbl['context_shift_date'])
    
    click_tbl_dict = {}
    for item, record in click_tbl.iterrows():
        click_tbl_dict[item] = record.tolist()
    
    smoothed_cvrt = {}
    alpha_val = {}
    beta_val = {}
    
    for idx, (k, count) in enumerate(impression_tbl_dict.items()):
        if not idx % 1000:
            print("processing {}: {}".format(col, idx))
        
        bs = BayesianSmoothing(1, 1)
        
        I = count[:7]
        C = click_tbl_dict.get(k, [0] * 7)
        
        bs.update(I, C, 1000, 0.0000000001)
        
        ctr = []
        for i in range(len(I)):
            ctr.append((C[i] + bs.alpha) / (I[i] + bs.alpha + bs.beta))
        
        smoothed_cvrt[k] = ctr
        alpha_val[k] = bs.alpha
        beta_val[k] = bs.beta
    
    smoothed_cvrt_tbl = pd.DataFrame(smoothed_cvrt).T.unstack().reset_index().dropna()
    smoothed_cvrt_tbl.columns = ['context_shift_date', col, '{}_smoothed_cvrt'.format(col)]
    
    if (col == 'item_id') or (col == 'item_brand_id'):
        save_cache(smoothed_cvrt_tbl, "{}_cvrt".format(col))
    
    smoothed_cvrt_tbl['context_shift_date'] = smoothed_cvrt_tbl['context_shift_date'] + 1
    
    # cvrt_beta_alpha = pd.concat([pd.Series(alpha_val), pd.Series(beta_val)], axis=1).dropna()
    # cvrt_beta_alpha.columns = ['{}_alpha'.format(col), '{}_beta'.format(col)]
    
    # save_cache(smoothed_cvrt_tbl, "{}_smoothed_cvrt".format(col))
    
    output = dfAll.loc[:, ['item_category_list2', 'instance_id', 'context_shift_date', col]].merge(smoothed_cvrt_tbl,
                                                                                                   how='left')
    
    output['{}_smoothed_cvrt'.format(col)] = \
        output.groupby(['item_category_list2', 'context_shift_date'])['{}_smoothed_cvrt'.format(col)].transform(lambda
                                                                                                                    x:
                                                                                                                x.fillna(
                                                                                                                    x.mean()))
    output = output.loc[:, ['instance_id', '{}_smoothed_cvrt'.format(col)]]
    
    return output


# target_cols = ['item_id', 'item_brand_id', 'item_city_id', 'item_sales_level', 'item_price_level', 'item_pv_level',
#                'context_page_id', 'context_shift_hour'] + [ "{}_qbin".format(s) for s in score_cols]


target_cols = ['item_id', 'item_brand_id']

pool = mp.Pool(20)
out = pool.starmap(smoothed_conversion, [(dfAll, col) for col in target_cols])

# item_smoothed_cvrt = reduce(lambda x,y: pd.merge(x, y, how='left'), out)

# save_valid(item_smoothed_cvrt, "item_smoothed_cvrt")
# save_test(item_smoothed_cvrt, "item_smoothed_cvrt")

pool.close()
