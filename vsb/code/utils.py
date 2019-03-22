import numpy as np
import math
from tqdm import tqdm
import pyarrow.parquet as pq
import pandas as pd
import pywt
from scipy import signal, stats
from tsfresh.feature_extraction.feature_calculators import *


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def high_pass_filter(x, low_cutoff=1000, sample_rate=50 * 800000):
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    sos = signal.butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)
    return filtered_sig


def wavelet_denoise(x, wavelet='db1', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


def signal_entropy(y):
    for i in range(3):
        max_pos = y.argmax()
        y[max_pos - 1000:max_pos + 1000] = 0.

    return stats.entropy(np.histogram(y, 15)[0])


def detail_coeffs_entropy(x, wavelet='db1'):
    c_a, c_d = pywt.dwt(x, wavelet)

    return stats.entropy(np.histogram(c_d, 15)[0])


def bucketed_entropy(x):
    y = wavelet_denoise(x)

    return np.array([stats.entropy(np.histogram(bucket, 10)[0]) for bucket in np.split(y, 10)])


def peaks(x):
    y = wavelet_denoise(x)
    peaks, properties = signal.find_peaks(y)
    widths = signal.peak_widths(y, peaks)[0]
    prominences = signal.peak_prominences(y, peaks)[0]
    return {
        'count': peaks.size,
        'width_mean': widths.mean() if widths.size else -1.,
        'width_max': widths.max() if widths.size else -1.,
        'width_min': widths.min() if widths.size else -1.,
        'prominence_mean': prominences.mean() if prominences.size else -1.,
        'prominence_max': prominences.max() if prominences.size else -1.,
        'prominence_min': prominences.min() if prominences.size else -1.,
    }


def feature(ts, n_dim=160):
    # https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/80166
    ts = high_pass_filter(ts, low_cutoff=10000, sample_rate=50 * 800000)
    ts_std = wavelet_denoise(ts)
    bucket_size = int(800000 / n_dim)
    new_ts = []
    for i in range(0, 800000, bucket_size):
        ts_range = ts_std[i:i + bucket_size]

        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 30, 50, 60, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]
        kurt = stats.kurtosis(ts_range)
        entropy = signal_entropy(ts_range)
        de_coeffs_entropy = detail_coeffs_entropy(ts_range)
        bucketed_entropy_ = bucketed_entropy(ts_range)
        peak = peaks(ts_range)
        c = peak['count']
        width_mean = peak['width_mean']
        width_max = peak['width_max']
        width_min = peak['width_min']
        prominence_mean = peak['prominence_mean']
        prominence_max = peak['prominence_max']
        prominence_min = peak['prominence_min']
        relative_percentile = percentil_calc - mean
        complexity = cid_ce(ts_range, normalize=True)
        nonlinear = c3(ts_range, lag=50)

        new_ts.append(
            np.concatenate(
                [np.asarray([mean, std, std_top, std_bot, max_range, entropy,
                             kurt, de_coeffs_entropy, c, width_mean, width_max,
                             width_min, prominence_mean, prominence_max, prominence_min,
                             complexity, nonlinear]),
                 percentil_calc,
                 relative_percentile,
                 bucketed_entropy_,
                 ]))
    return np.asarray(new_ts)


def get_score(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_pred_pos = np.round(y_pred)
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(y_true)
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)

    return (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-15)


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    raw_score = get_score(y_true.astype(np.float64), (y_proba > 0.5).astype(np.float64))
    for th in np.linspace(0, 1, 100):
        score = get_score(y_true.astype(np.float64), (y_proba > th).astype(np.float64))
        if score > best_score:
            best_threshold = th
            best_score = score

    return best_threshold, best_score, raw_score


def preprocess(df):
    if 'target' not in df.columns:
        df['target'] = np.zeros(len(df))

    def flatten(data):
        id_measurement = data['id_measurement'].values[0]
        target = f"{data['target'].values[0]}{data['target'].values[1]}{data['target'].values[2]}"
        record = [id_measurement, target] + data['signal_id'].tolist()
        return pd.Series(record, index=['id_measurement', 'target', 'signal_id1', 'signal_id2', 'signal_id3'])

    df = df.sort_values(by=['id_measurement', 'phase'], ascending=True)[['signal_id', 'id_measurement', 'target']]
    df = df.groupby(by=['id_measurement']).apply(flatten)
    return df


def worker(data, dim):
    r = []
    for x in data:
        y = feature(x, n_dim=dim)
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
        np.save(f'../data/downsampling_{dim}_{f}_signal.npy', results)

    fun(train=True, dim=dim)
    fun(train=False, dim=dim)


if __name__ == '__main__':
    # downsample(dim=160)
    t = np.load('../data/downsampling_160_train_signal.npy')
    print(t.shape)
    df = pd.read_csv('../data/metadata_train.csv')
    df = preprocess(df)
    print(df)
    # x = np.random.randn(100)
