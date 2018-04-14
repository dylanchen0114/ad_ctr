# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""
import numpy as np


def get_group(df, cols):
    group = df[cols[0]].copy()
    for col in cols[1:]:
        group = group.astype(str) + '_' + df[col].astype(str)

    return group


def count(df, cols):
    group = get_group(df, cols)
    count_map = group.value_counts()

    return group.map(count_map).fillna(0)


def unique_entity_cnt(df, cols, entity):
    group = get_group(df, cols)
    group_unique_entity = df.groupby(group)[entity].nunique()

    result = group.map(group_unique_entity)

    return result


def count_from_future(df, cols):
    result = []
    df_reverse = df.sort_values('context_timestamp', ascending=False)
    group = get_group(df_reverse, cols)

    count = {}
    for g in group.values:
        if g in count:
            result.append(count[g])
            count[g] += 1
        else:
            result.append(0)
            count[g] = 1

    result.reverse()
    return result


def count_from_past(df, cols):
    group = get_group(df, cols)

    count = {}
    result = []
    for g in group.values:
        if g not in count:
            count[g] = 0
        else:
            count[g] += 1
        result.append(count[g])

    return result


def last_time_diff(df, cols):
    group = get_group(df, cols)

    last_time = df.groupby(group).context_timestamp.last()

    return np.log1p(group.map(last_time) - df.context_timestamp)


def time_from_prev_expose(df, cols):
    group = get_group(df, cols)

    last_expose = df.groupby(group).context_timestamp.shift(1)

    return np.log1p(df.context_timestamp - last_expose)


def time_to_next_expose(df, cols):
    result = []
    df_reverse = df.sort_values('context_timestamp', ascending=False)
    group = get_group(df_reverse, cols)

    next_heard = {}
    for g, t in zip(group, df_reverse.context_timestamp):
        if g in next_heard:
            result.append(t - next_heard[g])
        else:
            result.append(-1)
        next_heard[g] = t

    result.reverse()
    return result


