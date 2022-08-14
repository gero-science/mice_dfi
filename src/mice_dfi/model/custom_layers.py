# -*- coding: utf8 -*-
"""
Custom layers includes: Inverse gradient layer, Linear scaling,
"""
import numpy
import scipy
import tensorflow as tf
import tensorflow.keras.layers as tf_layers


class Dense_AR_base(tf_layers.Layer):
    def __init__(self, rank, inner, **kargs):
        super().__init__(**kargs)
        self.rank = int(rank)
        self.inner = int(inner)
        self.ssi_A = None
        self.ssi_B = None
        self.eigvals = None
        self.weight_constrain = 100.

    def constraint_matrices(self):
        # Constraint for spectral decomposition : |A*B - I|_F -> min
        loss_ab = tf.matmul(self.ssi_A, self.ssi_B, transpose_a=True, transpose_b=True)
        loss_ab -= tf.eye(self.rank, dtype=tf.keras.backend.floatx())
        loss_ab = tf.reduce_mean(tf.square(loss_ab))
        return self.weight_constrain * loss_ab

    def _init_vars(self):
        floatx = tf.keras.backend.floatx()
        S = numpy.random.randn(self.inner, self.inner)
        S = numpy.dot(S.T, S)
        _l, av, bv = scipy.linalg.eig(S, left=True, right=True)
        av = numpy.conj(av.T)
        av /= numpy.diag(numpy.dot(av, bv))

        init_A = av[:self.rank].astype(floatx)
        init_B = bv[:, :self.rank].astype(floatx)
        init_eig = numpy.random.uniform(0.1, 0.5, size=self.rank).astype(floatx)
        init_eig[0] = 0.99
        return init_eig, init_A, init_B

    def get_config(self):
        config = {'rank': self.rank, 'inner': self.inner}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class Dense_AR(Dense_AR_base):

    def build(self, input_shape):

        assert self.inner == input_shape[1]

        init_eig, init_A, init_B = self._init_vars()

        self.ssi_A = self.add_weight(
            name='ssi_A', shape=(self.inner, self.rank),  trainable=True,
            initializer=tf.constant_initializer(init_A),
        )

        self.ssi_B = self.add_weight(
            name='ssi_B', shape=(self.rank, self.inner,),  trainable=True,
            initializer=tf.constant_initializer(init_B),
            constraint=tf.keras.constraints.UnitNorm(axis=1)
        )

        self.eigvals = self.add_weight(
            name='eigenvals', shape=(self.rank,), initializer=tf.constant_initializer(init_eig), trainable=True
        )

        self.shift_z = self.add_weight(
            name='shift', shape=(self.rank,), initializer='zeros', trainable=True,
        )

        self.nonlin = self.add_weight(  # deprecated
            name='ssi_nonlin_S', shape=(self.rank, self.rank,),  trainable=False,
            initializer=tf.constant_initializer(0),
        )

        self.add_loss(self.constraint_matrices)

        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        is_left = kwargs.pop('is_left')
        res_z_a = tf.matmul(x, self.ssi_A, name='mult_z_a')
        if is_left:
            res_z_a = res_z_a*self.eigvals + self.shift_z
        res_z_b = tf.matmul(res_z_a, self.ssi_B, name='mult_z_b')
        return [res_z_a, res_z_b]


class Dense_AR_sex(Dense_AR):
    def build(self, input_shape):
        self.f_sex = self.add_weight(
                name='f_sex', shape=(1,),  trainable=True,
                initializer=tf.constant_initializer(0.),
            )
        super().build(input_shape[0])  # Be sure to call this at the end

    def call(self, input_, **kwargs):
        x, sex = input_
        is_left = kwargs.pop('is_left')
        res_z_a = tf.matmul(x, self.ssi_A, name='mult_z_a')
        if is_left:
            res_z_a = res_z_a * self.eigvals + self.shift_z + self.f_sex[0] * sex
        res_z_b = tf.matmul(res_z_a, self.ssi_B, name='mult_z_b')
        return [res_z_a, res_z_b]


class Dense_out(tf.keras.layers.Dense):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def build(self, *args):
        super().build(*args)  # Be sure to call this at the end
        self.add_weight(self.name + '/w_ae', (), trainable=False, initializer=tf.initializers.ones)
        self.add_weight(self.name + '/w_ssi', (), trainable=False, initializer=tf.initializers.ones)
        self.add_weight(self.name + '/w_dan', (), trainable=False, initializer=tf.initializers.ones)


class Identity(tf_layers.Layer):

    def call(self, inputs, *args, **kwargs):
        return inputs

