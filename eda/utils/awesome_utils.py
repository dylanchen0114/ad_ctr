# -*- coding: utf-8 -*-
"""
@author: Xiaolan Zhu <xiaolan.zhu7@outlook.com>

"""

import seaborn as sns
import numpy as np

def cat_target_mean(data, cat_var, target='target', est=np.mean, aes_param=None):
    default_aes_param = {
        'errcolor': 'black',
        'errwidth': 0.5,
        'capsize': 0.08,
        'alpha': 0.9,
    }
    
    bar_aes_param = default_aes_param.copy()
    if aes_param:
        bar_aes_param.update(aes_param)
    
    ax = sns.countplot(cat_var, data=data, color="#f2552c", alpha=0.97)
    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.twinx()
    ax2 = sns.pointplot(x=cat_var, y=target, data=data, estimator=est, markers='o', join=False, scale=0.5,
                        color="#34495e", **bar_aes_param)
    ax2.set(ylabel='target mean')
    ax2.grid(False)
