import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import *
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

seed = 2341
random.seed(seed)
os.environ['PYTHONHASHSEED'] = f'{seed}'
np.random.seed(seed)

from sklearn.model_selection import StratifiedKFold

from utils import *


def train(model, df, fold, cfg):
    tr_x, tr_y, val_x, val_y = df

    tr_x = [tr_x[:, :, :45], tr_x[:, :, 45 * 1:45 * 2], tr_x[:, :, 45 * 2:]]
    val_x = [val_x[:, :, :45], val_x[:, :, 45 * 1:45 * 2], val_x[:, :, 45 * 2:]]

    best_epoch = 0
    best_pred = np.zeros_like(val_y, dtype=np.float64)
    best_score = -1

    for epoch in range(1000):
        if epoch - best_epoch > 5:
            break
        model.fit(tr_x,
                  tr_y,
                  batch_size=cfg['bs'],
                  epochs=1,
                  verbose=0,
                  )
        val_pred = model.predict(val_x)

        score = get_score(val_y, val_pred)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_pred = val_pred
            print(f'{epoch} score improved to {score}')
            model.save_weights(f"../model/{cfg['name']}{fold}.h5")
        else:
            print(f'{epoch} score {score} not improved, best {best_score}...........')

    return best_pred, best_score


def main(cfg):
    signal = np.load(f"../data/downsampling_{cfg['dim']}_train_signal.npy")

    tr = preprocess(pd.read_csv('../data/metadata_train.csv'))
    X = []
    Y = []
    prob = np.load('../data/adversarial.npy').reshape(-1)

    for item in tqdm(tr[['target', 'signal_id1', 'signal_id2', 'signal_id3']].values):
        s1 = signal[item[1]]
        s2 = signal[item[2]]
        s3 = signal[item[3]]
        X.append(np.concatenate([s1, s2, s3], axis=-1))
        label = [int(x) for x in item[0]]
        Y.append(label)

    X = np.asanyarray(X)
    Y = np.asanyarray(Y)

    assert len(prob) == len(X) == len(Y)

    nfolds = list(StratifiedKFold(n_splits=5, random_state=seed).split(X, Y[:, 0]))
    oofp = np.zeros_like(Y, dtype=np.float64)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(nfolds):
        K.clear_session()
        print("Beginning fold {}".format(fold + 1))
        model = rnn(cfg)

        val_idx = [x for x in val_idx if prob[x] > 0.5]

        print(len(val_idx))

        val_pred, score = train(model, (X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]), fold, cfg)

        oofp[val_idx] = val_pred

        scores.append(score)

    np.save('Y.npy', Y)
    np.save('oofp.npy', oofp)

    print(np.mean(scores))
    valid_idx = oofp > 0
    best_threshold, best_score, raw_score = threshold_search(Y[valid_idx], oofp[valid_idx])
    print(f'th {best_threshold}, val raw_score {raw_score}, val best score:{best_score}')


if __name__ == '__main__':
    from models import *

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

    print(cfg)
    main(cfg)

"""
# unit dp
{'name': 'rnn', 'dim': 160, 'channel': 24, 'unit1': 256, 'unit2': 64, 'unit3': 6
4, 'dp': 0.1, 'bn': True, 'lr': 0.001, 'bs': 32, 'alpha': 1.0, 'momentum': 0.99}  0.7109
{'name': 'rnn', 'dim': 160, 'channel': 24, 'unit1': 256, 'unit2': 128, 'unit3': 
384, 'dp': 0.1, 'bn': True, 'lr': 0.001, 'bs': 32, 'alpha': 1.0, 'momentum': 0.9   0.70
9}

# bs lr bn momentum
{'name': 'rnn', 'dim': 160, 'channel': 24, 'unit1': 128, 'unit2': 128, 'unit3': 
64, 'dp': 0.1, 'bn': True, 'lr': 0.0005, 'bs': 32, 'alpha': 1.0, 'momentum': 0.7
}

#alpha

1-1.15

"""
