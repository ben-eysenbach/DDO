import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import scipy.special

class BetaRegularization(Layer):

    def __init__(self, alpha, beta, **kwargs):
        super(BetaRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.beta = beta
        self._log_normalizer = scipy.special.betaln(alpha, beta)
        self.activity_regularizer = self._beta_regularizer

    def _beta_regularizer(self, activity_matrix):
        term_1 = (self.alpha - 1.0) * activity_matrix[..., 1] + (self.beta - 1.0) * activity_matrix[..., 0]
        term_2 = (self.alpha + self.beta - 2.0) * K.log(K.exp(activity_matrix[..., 0]) + K.exp(activity_matrix[..., 1]))
        log_prob = K.sum(term_1 - term_2 - self._log_normalizer)
        return -log_prob

    def get_config(self):
        config = {'alpha': self.alpha,
                  'beta': self.beta}
        base_config = super(BetaRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def test_beta_regularizer():
    alpha = 2.
    beta = 7.
    from tensorflow.keras.layers import Dense, Input, Softmax
    x = Input(shape=(1,), name='x')
    fc = Dense(2, activation=None)(x)
    fc = BetaRegularization(alpha, beta)(fc)
    p = Softmax(axis=-1)(fc)
    model = tf.keras.Model(inputs=[x], outputs=[p])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=['mean_squared_error'],
                  loss_weights=[0.0])
    model.fit([np.array([[0.0]])],
              [np.array([[0.0, 0.0]])],
              steps_per_epoch=1000,
              epochs=10,
              verbose=1)
    p = model.predict(np.array([[0.0]]))[0, 1]
    print('p = %.3g (vs %.3g)' % (p, (alpha / (alpha + beta))))


if __name__ == '__main__':
    test_beta_regularizer()
