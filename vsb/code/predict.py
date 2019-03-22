# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict
   Description :
   Author :       haxu
   date：          2019/2/25
-------------------------------------------------
   Change Activity:
                   2019/2/25:
-------------------------------------------------
"""
__author__ = 'haxu'

import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

import numpy as np
from utils import preprocess, tqdm
from models import rnn


def main(cfg):
    te = preprocess(pd.read_csv('../data/metadata_test.csv'))

    signal_data = np.load(f"../data/downsampling_{cfg['dim']}_test_signal.npy")
    signal_data = np.concatenate([np.zeros((8712, cfg['dim'], cfg['channel'])), signal_data], axis=0)  # 注意axis

    X = []
    for item in tqdm(te[['signal_id1', 'signal_id2', 'signal_id3']].values.astype(np.int32)):
        s1 = signal_data[item[0]]
        s2 = signal_data[item[1]]
        s3 = signal_data[item[2]]
        X.append(np.concatenate([s1, s2, s3], axis=-1))
    X = np.asanyarray(X)
    X = [X[:, :, :45 * 1], X[:, :, 45 * 1:45 * 2], X[:, :, 45 * 2:]]

    r = 0
    for w in ['rnn0.h5', 'rnn1.h5', 'rnn2.h5', 'rnn3.h5', 'rnn4.h5']:
        model = rnn(cfg)
        model.load_weights(f'../model/{w}')
        y = model.predict(X, batch_size=1024, verbose=1)
        r += y

    y = (r / 5 > cfg['best_threshold']).astype(np.float64).tolist()

    y = [int(x) for a in y for x in a]

    print(len(y))

    df = pd.read_csv('../data/metadata_test.csv')
    df['target'] = y

    df[['signal_id', 'target']].to_csv('res.csv', index=False)
    print('done.....')


if __name__ == '__main__':
    cfg = {}
    cfg['name'] = 'rnn'
    cfg['dim'] = 160
    cfg['channel'] = 45
    cfg['unit1'] = 256
    cfg['unit2'] = 128
    cfg['unit3'] = 384
    cfg['dp'] = 0.1
    cfg['bn'] = True
    cfg['lr'] = 1e-3
    cfg['bs'] = 32

    cfg['alpha'] = 1.
    cfg['momentum'] = 0.99

    cfg['best_threshold'] = 0.5

    print(cfg)
    main(cfg)
