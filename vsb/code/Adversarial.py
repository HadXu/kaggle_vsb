# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Adversarial
   Description :
   Author :       haxu
   date：          2019/3/5
-------------------------------------------------
   Change Activity:
                   2019/3/5:
-------------------------------------------------
"""
__author__ = 'haxu'

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

from utils import *
from models import adversarial


def extract_features(ts, n_dim=160):
    ts = high_pass_filter(ts, low_cutoff=10000, sample_rate=50 * 800000)
    ts_std = wavelet_denoise(ts)
    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        ts_range = ts_std[i:i + bucket_size]

        max = ts_range.max()
        min = ts_range.min()
        ptp = max - min
        new_ts.append(ptp)
    return np.asarray(new_ts)


def worker(data, dim):
    r = []
    for x in tqdm(data):
        y = extract_features(x, n_dim=dim)
        r.append(y)
    return r


def downsample(dim=160):
    import multiprocessing as mp

    def fun(train=True, dim=160):
        f = 'train'
        if not train:
            f = 'test'
        signal_data = pq.read_pandas(f'../data/{f}.parquet').to_pandas().values.T
        pool = mp.Pool()
        num_worker = mp.cpu_count()
        num_samples = len(signal_data)
        avg = 1 + num_samples // num_worker
        res = []
        for i in range(num_worker):
            result = pool.apply_async(worker, args=(signal_data[i * avg:(i + 1) * avg], dim))
            res.append(result)
        pool.close()
        pool.join()

        results = []

        for r in res:
            results += r.get()
        results = np.asanyarray(results)
        np.save(f'../data/adversarial_{dim}_{f}_signal.npy', results)

    fun(train=True, dim=dim)
    fun(train=False, dim=dim)


def main():
    # downsample()
    tr_feat = np.load('../data/downsampling_160_train_signal.npy')
    te_feat = np.load('../data/downsampling_160_test_signal.npy')

    num_train = 8712
    num_test = 20337
    tr = []
    for i in range(0, num_train, 3):
        tr.append([tr_feat[i], tr_feat[i + 1], tr_feat[i + 2]])
    te = []
    for i in range(0, num_test, 3):
        te.append([te_feat[i], te_feat[i + 1], te_feat[i + 2]])

    tr = np.asanyarray(tr)
    te = np.asanyarray(te)

    X = np.concatenate((tr, te))  # (9683, 3, 160, 45)

    y = np.zeros((num_train // 3 + num_test // 3))
    y[num_train // 3:num_train // 3 + num_test // 3] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=32)

    tr_x = [X_train[:, 0], X_train[:, 1], X_train[:, 2]]
    val_x = [X_test[:, 0], X_test[:, 1], X_test[:, 2]]
    model = adversarial()

    for e in range(10):
        model.fit(tr_x, y_train, verbose=1, epochs=1)
        y_pred = model.predict(val_x).reshape(-1)

        auc = roc_auc_score(y_test, y_pred)

        print(f'{e}   --  auc{auc}.......')

        model.save_weights('../model/adv_classifier.h5')


def gen_train_val():
    model = adversarial()
    model.load_weights('../model/adv_classifier.h5')
    tr_feat = np.load('../data/downsampling_160_train_signal.npy')

    num_train = 8712
    tr = []
    for i in range(0, num_train, 3):
        tr.append([tr_feat[i], tr_feat[i + 1], tr_feat[i + 2]])
    tr = np.asanyarray(tr)
    tr_x = [tr[:, 0], tr[:, 1], tr[:, 2]]

    y_liketest = model.predict(tr_x)
    y_sorted = np.sort(y_liketest, axis=0)

    eighty = int(num_train // 3 * .8)
    twenty = num_train // 3 - eighty

    thresh80 = y_sorted[eighty, 0]  # 0.743947

    print(thresh80)

    np.save('../data/adversarial.npy', y_liketest)


if __name__ == '__main__':
    # main()

    gen_train_val()