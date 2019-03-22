from keras.layers import *
from keras.optimizers import *
from keras.models import Model, Sequential


def matthews_correlation(y_true, y_pred):
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    #
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# sdp   dp  fc unit  lr  bs


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def rnn(cfg):
    def get_loss(y_true, y_pred):
        loss = cfg['alpha'] * y_true * K.log(y_pred) + \
               (1 - y_true) * K.log(1 - y_pred) * (2 - cfg['alpha'])
        return -K.mean(loss)

    # inp = Input(shape=(cfg['dim'], cfg['channel'] * 3))
    #
    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    # x = Attention(cfg['dim'])(x)
    # x = Dense(64, activation="relu")(x)
    # x = Dense(1, activation="sigmoid")(x)
    s1 = Input((cfg['dim'], cfg['channel']), name='s1')
    s2 = Input((cfg['dim'], cfg['channel']), name='s2')
    s3 = Input((cfg['dim'], cfg['channel']), name='s3')
    inp = [s1, s2, s3]

    x = [s1, s2, s3]
    lstm1 = Bidirectional(CuDNNLSTM(cfg['unit1'], return_sequences=True))
    lstm2 = Bidirectional(CuDNNLSTM(cfg['unit2'], return_sequences=True))
    for i in range(3):
        if cfg['bn']:
            x[i] = BatchNormalization(momentum=cfg['momentum'])(x[i])
        x[i] = lstm1(x[i])
        x[i] = lstm2(x[i])
        x[i] = GlobalAveragePooling1D()(x[i])
    context = average(x)
    fc1 = Dense(cfg['unit3'], activation="relu")
    for i in range(3):
        x[i] = concatenate([x[i], context])
        x[i] = Dropout(cfg['dp'])(x[i])
        x[i] = fc1(x[i])
        x[i] = Dense(1, activation="sigmoid")(x[i])

    x = concatenate(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss=get_loss,
                  optimizer=Nadam(cfg['lr']),
                  metrics=[matthews_correlation]
                  )

    return model


def adversarial():
    dim = 160
    channel = 45
    s1 = Input((dim, channel), name='s1')
    s2 = Input((dim, channel), name='s2')
    s3 = Input((dim, channel), name='s3')
    inp = [s1, s2, s3]

    x = [s1, s2, s3]
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))
    lstm2 = Bidirectional(CuDNNLSTM(64, return_sequences=True))
    for i in range(3):
        x[i] = lstm1(x[i])
        x[i] = lstm2(x[i])
        x[i] = GlobalAveragePooling1D()(x[i])

    x = concatenate(x)
    x = Dense(32)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'],
                  )
    return model


# def rnn1(cfg):
#     s1 = Input((cfg['num_steps'], 24), name='s1')
#     s2 = Input((cfg['num_steps'], 24), name='s2')
#     s3 = Input((cfg['num_steps'], 24), name='s3')
#
#
#
#
#
#     s = concatenate([s1, s2, s3], axis=-1)
#     s = BatchNormalization()(s)
#
#     s = SpatialDropout1D(0.1)(s)
#
#     x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(s)
#     x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
#     x = GlobalAveragePooling1D()(x)
#
#     x = Dense(128, activation="relu")(x)
#     x = Dense(64, activation="relu")(x)
#     x = Dropout(0.1)(x)
#     output = Dense(1, activation="sigmoid")(x)
#
#     model = Model(inputs=[s1, s2, s3], outputs=output)
#     model.compile(loss='binary_crossentropy',
#                   optimizer=Nadam(lr=cfg['lr']),
#                   metrics=[matthews_correlation],
#                   )
#
#     return model


if __name__ == '__main__':
    cfg = {}
    cfg['dim'] = 160
    cfg['lr'] = 1e-3
    cfg['channel'] = 3
    model = adversarial()
    model.summary()
