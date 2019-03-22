# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     vote
   Description :
   Author :       haxu
   date：          2019/3/7
-------------------------------------------------
   Change Activity:
                   2019/3/7:
-------------------------------------------------
"""
__author__ = 'haxu'

import pandas as pd
import os
from scipy import stats

csvs = [x for x in os.listdir('../data/') if x.endswith('.csv')]

df = pd.DataFrame({}, index=range(8712, 29049))

csvs = ['709.csv', '705.csv']

for i, f in enumerate(csvs):
    t = pd.read_csv(f'../data/{f}', index_col=0)
    df[f'{i}_target'] = t['target'].values

x = stats.mode(df[[f'{i}_target' for i in range(len(csvs))]].values, axis=1)

df.index.name = 'signal_id'
df['target'] = x[0].reshape(-1)

df[['target']].to_csv('res.csv')
