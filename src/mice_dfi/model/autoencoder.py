# -*- coding: utf8 -*-
"""
Autoencoder + SSI with tensorflow and keras
"""

import inspect
import os
import pickle
import collections
import re
import json
import random
import copy

import tensorflow as tf
import numpy as np
import sklearn.model_selection as SKMS

from . import custom_layers
from .blocks import BaseArchitecture, ResNetAE


class BaseAutoencoder(BaseArchitecture):
    _bottleneck_name = 'bottleneck'
    _input_name = 'in_ae'
    _in_ssi_x_name = 'in_ssi_x'
    _in_ssi_y_name = 'in_ssi_y'
    _decoder_name = 'decoder_w/o_noise'
    _out_AE_name = 'autoencoder'
    _out_ssi_name = 'merged_linear_ssi'
    _out_ssi_raw_name = 'en_dec_ssi_to_raw'

    def __init__(self, rank=None, input_dim=None, load=None, **options):
        """ simple transcoder X[n] -> X[n+1]

        Parameters
        ----------
        rank : int,
            size of the bottleneck
        input_dim : int
            size of the input data
        """

        self.input_dim = input_dim
        self.inner_dim = rank
        self.rank = rank
        self.config = {}

        # Data processing
        self.mean_ = None
        self.std_ = None
        self.preproc = None
        self.scaling = 'stand'
        self.nepoch = 0
        self.history = {}

        seed = options.pop('seed', None)
        if seed is None:
            seed = np.random.randint(2 ** 30)
        self.seed = seed
        self._set_seeds()
        self._rnd = np.random.RandomState(self.seed)

        # Set predefine settings
        self._set_keras_config(**options)

        self.model = None
        self.model_encoder = None
        self.model_autoencoder = None

        self._saver = None
        self._loss_weights = dict()
        self._loss_weights_names = []
        self._feeder_train = None
        self._feeder_test = None
        tf.keras.backend.set_floatx(self.config['floatX'])

    @staticmethod
    def tf_r2(y_true, y_pred):
        total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
        return 1. - unexplained_error / total_error

    @staticmethod
    def tf_r2_modified(y_true, y_pred):
        error = y_true - y_pred
        error = tf.square(error)
        error /= tf.math.reduce_variance(y_true, axis=0)
        error = tf.reduce_mean(error)
        return error

    @staticmethod
    def score_r2(x_true, x_pred):
        total_error = np.sum(np.square(x_true - np.mean(x_true, axis=0)))
        unexplained_error = np.sum(np.square(x_true - x_pred))
        return 1. - unexplained_error / total_error

    @staticmethod
    def loss_mse(x_true, x_pred):
        return np.square(x_pred - x_true).sum() / x_pred.size

    @staticmethod
    def loss_lmse(x_true, x_pred):
        r2 = np.square(x_true - x_pred) * 0.5
        r2 = np.mean(r2, axis=1)
        loss = np.log(1. + r2)
        return np.mean(loss)

    @staticmethod
    def correlation(x, y):
        return np.mean((x - x.mean(axis=0)) * (y - y.mean(axis=0)), axis=0) / np.std(x, axis=0) / np.std(y, axis=0)

    def get_default_loss(self, data, steps=None):
        """
        Returns loss calculated by keras model

        """
        return self.model.evaluate_generator(data, verbose=0, steps=steps)

    def _build_model(self):
        raise NotImplementedError

    @staticmethod
    def get_model_var(tf_var):
        """ Get tensor value used in model """
        return tf.keras.backend.get_value(tf_var)

    def get_ae_loss(self, x_true, scale=True, loss='mse'):
        x_pred = self.predict_decoded(x_true, scale=scale)
        if loss == 'mse':
            return self.loss_mse(x_true, x_pred)
        elif loss == 'lmse':
            return self.loss_lmse(x_true, x_pred)

    def get_ae_score(self, x_true, scale=True):
        x_pred = self.predict_decoded(x_true, scale=scale)
        return self.score_r2(x_true, x_pred)

    def predict_decoded(self, *args, **kwargs):
        return np.empty(0)

    def scale(self, X, scale=True):
        if not scale or self.scaling == 'raw':
            return X
        X = X.copy()
        X = self._preproc_data(X, self.preproc, False)
        if self.scaling == 'stand':
            return (X - self.mean_) / self.std_
        else:
            raise ValueError('Unknown scaling %s' % self.scaling)

    def unscale(self, X, scale=True):
        if not scale or self.scaling == 'raw':
            return X
        X = X.copy()
        if self.scaling == 'stand':
            X = X * self.std_ + self.mean_
        else:
            X = X
        X = self._preproc_data(X, self.preproc, True)
        return X

    def _preproc_data(self, X, preproc, reverse=False):

        if preproc is None:
            return X

        if reverse:
            preproc = list(preproc)
            preproc = [w + 'r' for w in preproc]

        X = X.copy()
        for ii in range(X.shape[1]):
            if preproc[ii] == 'None' or preproc[ii] == 'Noner':
                pass
            elif preproc[ii] == 'log':
                X[:, ii] = np.log(X[:, ii])
            elif preproc[ii] == 'logr':
                X[:, ii] = np.exp(X[:, ii])
            else:
                ValueError('Unknown preprocessing function')

        return X

    @staticmethod
    def _shuffle_stratified_labels(labels, batch, fraction=0.85, seed=None):
        rnd = np.random.RandomState(seed)
        uvals, ucounts = np.unique(labels, return_counts=True, axis=0)
        m = np.asarray(ucounts >= batch)
        train_frac_max = ucounts[m].sum() * 1. / sum(ucounts)
        train_frac_min = np.sum(m) * batch * 1. / sum(ucounts)

        if train_frac_max < fraction or train_frac_min > fraction:
            raise RuntimeError('Impossible to train data with fraction=%.2f but split possible in %.3f - %.3f'
                               ' change batch size or train fraction.' %
                               (fraction, train_frac_min, train_frac_max))
        test_ind = []
        train_ind = []
        for i in range(len(ucounts)):
            ind = np.equal(labels, uvals[i])
            if labels.ndim > 1:
                ind = ind.all(axis=1)
            ind = np.argwhere(ind)
            if m[i]:
                to_train = np.random.choice(ind.flatten(),
                                            size=max(batch, int(len(ind.flatten()) * fraction)),
                                            replace=False)
                train_ind += to_train.tolist()
                test_ind += np.setdiff1d(ind.flatten(), to_train).tolist()
            else:
                test_ind += ind.flatten().tolist()

        train_ind, test_ind = np.array(train_ind, dtype=np.int32), np.array(test_ind, dtype=np.int32)
        max_iter = 1000
        ii = 0

        while np.abs(len(train_ind) * 1. / len(labels) - fraction) > 0.01:
            ii += 1
            balance_ind = int(len(labels) * (1 - fraction)) - len(test_ind) + 1
            if balance_ind > 0:
                select_from = train_ind
                add_to_test = True
            else:
                balance_ind = abs(balance_ind)
                select_from = test_ind
                add_to_test = False
                pass
            selected = rnd.choice(select_from, size=balance_ind)
            for s in selected:
                ind = np.equal(labels[train_ind], labels[s])
                if labels.ndim > 1:
                    ind = ind.all(axis=1)
                len_selected = sum(ind)
                if len_selected >= batch:
                    if add_to_test and len_selected > batch:
                        train_ind = np.setdiff1d(train_ind, np.int32(s))
                        test_ind = np.append(test_ind, np.int32(s))
                    elif add_to_test:  # Do not remove from train if batch is min
                        pass
                    else:
                        test_ind = np.setdiff1d(test_ind, np.int32(s))
                        train_ind = np.append(train_ind, np.int32(s))
            if ii % 100 == 0:
                print(ii, 'step', 'train frac', len(train_ind) * 1. / len(labels))
            if ii > max_iter:
                raise RuntimeError('Impossible to split data, decrease batch size or increase train fraction')
        return train_ind, test_ind

    def _data_prep(self, X, groups, train_ratio, seed, update_stats=True, return_index=False,
                   index_split=None, feature_weights=None, feature_preproc=None,
                   stratified_labels=None, batch=None):
        """ Prepare data for AE"""

        X = X.astype(self.config['floatX'])

        if not np.isfinite(X).all():
            raise RuntimeError("Not all values are finite")

        if feature_preproc is not None:
            assert len(feature_preproc) == X.shape[1]
            self.preproc = [str(w) for w in feature_preproc]

        if index_split is None:
            if groups is None and stratified_labels is None:
                data_split = SKMS.ShuffleSplit(n_splits=1, test_size=(1. - train_ratio), random_state=seed)
                train_ind, val_ind = next(data_split.split(X, y=None, groups=groups))
            elif stratified_labels is None:
                data_split = SKMS.GroupShuffleSplit(n_splits=1, test_size=(1. - train_ratio), random_state=seed)
                train_ind, val_ind = next(data_split.split(X, y=None, groups=groups))
            elif stratified_labels is not None:
                train_ind, val_ind = self._shuffle_stratified_labels(
                    stratified_labels, batch, train_ratio, seed=seed)
            else:
                raise ValueError

            X_train, X_test = X[train_ind], X[val_ind]
        else:
            X_train, X_test = X[index_split[0]], X[index_split[1]]

        if update_stats is True:
            print("Updating stat", self.preproc)
            Xstat = self._preproc_data(X, self.preproc)
            if not np.isfinite(Xstat).all():
                raise RuntimeError("Not all values are finite after preprocessing")
            self.mean_ = Xstat.mean(axis=0)
            self.std_ = Xstat.std(axis=0)
            if feature_weights is not None:
                assert len(feature_weights) == len(self.std_)
                assert (feature_weights <= 1).all() and (feature_weights > 0).all()
                self.std_ /= feature_weights

        X_train = self.scale(X_train)
        X_test = self.scale(X_test)

        if return_index:
            return X_train, X_test, (train_ind, val_ind,)
        else:
            return X_train, X_test

    def get_keras_config(self, **kwargs):
        """ Get copy of current config of the model and updated it from kwargs

        Parameters
        ----------
        **kwargs : dict

        Returns
        -------
        config : dict
            model configuration
        """
        assert len(self.config) > 0, "Set model configuration first"
        config = copy.deepcopy(self.config)
        len0 = len(config)
        config.update(**kwargs)
        len1 = len(config)
        if len0 != len1:
            raise RuntimeError("Configuration shape was changed. "
                               "You probably pass wrong keys. Current keys are %s" % config.keys())
        return config

    def _set_keras_config(self, **kwargs):
        """
        Read kwarg to extract keras config for building encoder-decoder
        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        params : tuple

        """
        # Defaults
        nodes = kwargs.pop('nodes')
        dropouts = kwargs.pop('dropouts')
        if nodes is None:
            nodes = []
            dropouts = []
        leaky_alpha = kwargs.pop('leaky_alpha', None)
        bn_first = kwargs.pop('bn_first', True)
        resnet = kwargs.pop('ResNet', False)
        res_config = kwargs.pop('Resnet_config', None)
        corrupt_input = kwargs.pop('corrupt_input', None)
        disable_bn = kwargs.pop('disable_bn', False)
        float_type = kwargs.pop('floatX', 'float32')
        r2_ssi_loss = kwargs.pop('r2_ssi_loss', False)
        dense_props = kwargs.pop('dense_props', dict())
        act_props = kwargs.pop('act_props', dict())
        drop_props = kwargs.pop('drop_props', dict())
        batch_props = kwargs.pop('batch_props', dict())

        # Sanity checks
        assert len(nodes) == len(dropouts), "Bad configuration"
        assert len(nodes) == len(dropouts), "Bad configuration"
        if resnet:
            assert len(nodes) == len(res_config), "Bad configuration"

        # Dense layers settings
        dense_props_defaults = dict(activation='relu', kernel_initializer='lecun_normal',
                                    use_bias=True, kernel_regularizer=None)
        dense_props_defaults.update(dense_props)
        if dense_props_defaults['kernel_regularizer'] is not None:
            l2_lambda = float(dense_props_defaults['kernel_regularizer'])
            dense_props_defaults['kernel_regularizer'] = tf.keras.regularizers.L2(l2_lambda)

        if leaky_alpha is None and dense_props_defaults['activation'] == 'leaky':
            raise ValueError("Provide leaky_alpha if using activation `leaky`")

        # ignore alpha if not leaky relu
        if leaky_alpha is not None and dense_props_defaults['activation'] != 'leaky':
            leaky_alpha = None

        # Wrap it into dictionary
        config = {}
        config['nodes'] = nodes
        config['dropouts'] = dropouts
        config['ResNet'] = resnet
        if config['ResNet']:
            assert res_config is not None, "Please provide resnet configuration"
        if resnet:
            config['Resnet_config'] = res_config
        else:
            config['Resnet_config'] = None

        config['bn_first'] = bn_first
        config['corrupt_input'] = corrupt_input
        config['disable_bn'] = disable_bn
        config['floatX'] = float_type
        config['r2_ssi_loss'] = r2_ssi_loss
        config['dense_props'] = dense_props_defaults
        config['dense_activation'] = dense_props_defaults['activation']

        # Activation props
        act_props_defaults = dict()
        act_props_defaults.update(act_props)
        if leaky_alpha is not None:
            act_props_defaults['alpha'] = leaky_alpha
        config['act_props'] = act_props_defaults

        # Dropouts
        drop_props_defaults = dict(drop_class='GaussianNoise')
        drop_props_defaults.update(drop_props)
        config['drop_props'] = drop_props_defaults

        # BatachNorm
        batch_props_defaults = dict(renorm=False)
        batch_props_defaults.update(batch_props)
        config['batch_props'] = batch_props_defaults

        self.config = config

    def update_keras_config(self, **kwargs):
        """ Updates model configuration """
        assert len(self.config) > 0, "Set model configuration first"
        for key, val in kwargs:
            if key not in self.config:
                raise KeyError("Parameter %s is not part of model config" % key)
            self.config[key] = val

    def _build_input(self):
        """ Create input for autoencder with optional noise"""
        with tf.name_scope('Inputs'):
            Data_ae_X = tf.keras.layers.Input(shape=(self.input_dim,), name=self._input_name)
            if self.config['corrupt_input'] is not None:
                input_ae = self._get_allocate_layer(
                    'in_corrupt', 'drop', None, drop=self.config['corrupt_input'], **self.config['drop_props']) \
                    (Data_ae_X)
                # tf.keras.layers.GaussianNoise(self.config['corrupt_input'], name='in_corrupt')(Data_ae_X)
            else:
                input_ae = Data_ae_X
        return Data_ae_X, input_ae

    def _build_encoder(self, input_, name='EC_', model=None, config=None, no_drops=False):
        """
        Parameters
        ----------
        input_ : tf.Variable or tf.keras.layer.Input
            input variable
        model : tf.keras.models.Model, optional
            If None, build connected layers according to config, if keras object is passed then use shared layers
        config : dict, optional
            If None, take model current config. If dictionary is passed then use it

        Returns
        -------
        encoder : tf.Variable
            Last variable in the connected graph

        """

        if config is None:
            config = self.get_keras_config()
            if no_drops:
                disabled_dropouts = [None] * len(self.config['nodes'])
                config['dropouts'] = disabled_dropouts

        with tf.name_scope('Encoder'):
            if self.config['ResNet']:
                encoded = ResNetAE._build_branch(input_, model=model, suff=name, rnd=self._rnd, **config)

            else:
                encoded = BaseArchitecture._build_branch(input_, model=model, suff=name, rnd=self._rnd, **config)

            if encoded.shape[1] > self.inner_dim:
                dense_props_mod = BaseArchitecture._set_seed_dense_initializer(self.config['dense_props'], self._rnd)
                dense_props_mod['activation'] = 'linear'
                bottleneck = self._get_allocate_layer('bottle_unscaled', 'dense', model=model, node=self.inner_dim,
                                                      **dense_props_mod)
            elif encoded.shape[1] == self.inner_dim:
                bottleneck = self._get_allocate_layer('bottle_unscaled', 'identity', model=model)
            else:
                raise ValueError('Bottle neck size %d is larger than previous input %d' %
                                 (self.inner_dim, encoded.shape[1]))

            encoder = bottleneck(encoded)
        return encoder

    def _build_decoder(self, input_, name='DC_', model=None, config=None):
        """  Build decoder part of autoencoder """

        if config is None:  # get default config
            config = self.get_keras_config()
            disabled_dropouts = [None] * len(self.config['nodes'])  # always disable dropouts
            config['dropouts'] = disabled_dropouts
            config['nodes'] = self.config['nodes'][::-1]  # Use nodes in reverse order
            if config['ResNet']:
                config['Resnet_config'] = self.config['Resnet_config'][::-1]  # Use resnet blocks in reverse order

        with tf.name_scope('Decoder'):
            if self.config['ResNet']:
                decoded = ResNetAE._build_branch(input_, model=model, suff=name, rnd=self._rnd, **config)
            else:
                decoded = BaseArchitecture._build_branch(input_, model=model, suff=name, rnd=self._rnd, **config)

            if decoded.shape[1] != self.input_dim:
                dense_props_mod = BaseArchitecture._set_seed_dense_initializer(self.config['dense_props'], self._rnd)
                dense_props_mod['activation'] = 'linear'
                output_layer = self._get_allocate_layer(
                    'layer_' + self._decoder_name, 'dense', model=model, node=self.input_dim,
                    **dense_props_mod)
            else:
                output_layer = self._get_allocate_layer('layer_' + self._decoder_name, 'identity', model)
            decoder = output_layer(decoded)
        return decoder

    def _extract_encoder(self):
        if self.model is None:
            raise RuntimeError("Create model first")
        data = tf.keras.layers.Input(shape=(self.input_dim,), name='input_encoder')
        encoded_data = self._build_encoder(data, model=self.model, no_drops=True)
        self.model_encoder = tf.keras.models.Model(inputs=data, outputs=encoded_data)

    def _extract_autoencoder(self):
        if self.model is None:
            raise RuntimeError("Create model first")
        data = tf.keras.layers.Input(shape=(self.input_dim,), name='input_autoencoder')
        encoded_data = self._build_encoder(data, model=self.model, no_drops=True)
        ae_model = self.model.get_layer(self._out_AE_name)
        if not issubclass(type(ae_model), tf.keras.models.Model):
            ae_model = self.model
        decoded_data = self._build_decoder(encoded_data, model=ae_model)
        self.model_autoencoder = tf.keras.models.Model(inputs=data, outputs=decoded_data)

    def _extract_model_parts(self):
        self._extract_encoder()
        self._extract_autoencoder()

    def _init_loss_weights(self):
        """
        Allocate tf variables for loss weights
        """
        # FIND ID for Keras variable
        lst = self.model.weights[0].name.split('/')[0]
        lst = re.findall('\d+$', lst)
        suffix = ''
        if len(lst) == 1:
            suffix = lst[0]
        elif len(lst) > 1:
            raise RuntimeError("I don't know how to get id from %s" % self.model.weights[0].name)

        for name in self._loss_weights_names:
            self._loss_weights[name] = tf.Variable(
                1.0, name='loss%s/%s' % (suffix, name), trainable=False, dtype=self.config['floatX'])

    def _set_loss_weights(self, weights):
        """
        Set values for tf variables from dictionary

        Parameters
        ----------
            weights: dict
        """
        for name, val in weights.items():
            tf.keras.backend.set_value(self._loss_weights[name], np.asarray(val, dtype=self.config['floatX']))

    def _set_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # tf.keras.backend.set_session(sess)

    def _reset_model_weights(self, seed):
        """ Reset weights for trainable layers """

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # TODO fix weight reinitialization
        # session = tf.keras.backend.get_session()
        # sess = tf.Session(graph=tf.get_default_graph())
        # tf.keras.backend.set_session(sess)
        # print(session.graph.seed)
        # for layer in self.model.layers:
        #     for v in layer.variables:
        #         v.initializer.run(session=session)

    @staticmethod
    def loss_ae(y_true, y_pred, weight=1.):
        return tf.keras.losses.mse(y_true, y_pred) * weight

    def _compile_model(self, optimizer=None):
        pass

    def set_optimizer(self, name, **props):
        optimizer = getattr(tf.keras.optimizers, name)
        optimizer = optimizer(**props)
        self._compile_model(optimizer)

    def train(self, *args, **kargs):
        """ Train prototype """
        pass

    def save_model(self, dirname, ave_hist=5):
        os.makedirs(dirname, exist_ok=True)
        print("Saving model at ", dirname)
        NN_path = os.path.join(dirname, 'NN.hdf5')
        meta_path = os.path.join(dirname, 'meta.npz')
        meta_path_pkl = os.path.join(dirname, 'meta.pkl')

        save_dict = {}
        for i, item in self.__dict__.items():
            if type(item) in [np.ndarray, type(None), bool, int, float, str, tuple, list, dict]:
                if i[0] == '_':
                    continue
                save_dict[i] = item

        # Save weights
        w = dict()
        if len(self._loss_weights) > 0:
            for name, val in self._loss_weights.items():
                w[name] = tf.keras.backend.get_value(val)
            save_dict['_loss_weights'] = w

        # Save metadata and configs
        np.savez(meta_path, **save_dict)
        with open(meta_path_pkl, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save keras model
        tf.keras.models.save_model(self.model, NN_path, overwrite=True, include_optimizer=False)

        # Save history
        with open(os.path.join(dirname, 'history.json'), 'w') as f:
            dict_ = self._get_history_last_ave(ave_hist)  # self.nn_model
            dict_['step'] = self.nepoch
            json.dump(dict_, f)
        return dirname

    def _save_every(self, base_dir, step):
        """ Save model every epoch equals `step` """
        if base_dir is None:
            return
        if step <= 0:
            return
        if self.nepoch % step != 0:
            return
        os.makedirs(base_dir, exist_ok=True)
        self.save_model(os.path.join(base_dir,  'step%d' % self.nepoch))

    def _set_history_collection(self):
        self._history_collection = collections.OrderedDict()
        self._history_args = collections.OrderedDict()

    @staticmethod
    def _append_to_dict(d, key, val, step):
        try:
            d[key][step] = val
        except KeyError:  # allocate history
            if hasattr(val, '__len__'):
                d[key] = np.zeros((100, len(val)), dtype=np.float32)
            else:
                d[key] = np.zeros(100, dtype=np.float32)
            d[key][step] = val
        except IndexError:  # grow history
            d[key] = np.concatenate((d[key], np.zeros_like(d[key], dtype=np.float32)))
            d[key][step] = val

    def _update_history(self, step):
        for loss, item in self._history_collection.items():
            f = self.__getattribute__(item)
            args = self._history_args[loss]
            data = f(*args)  # calc_history(col)
            self._append_to_dict(self.history, loss, data, step)

    def _get_history_last_ave(self, ave_last=5):
        dict_ = dict()
        for key, val in self.history.items():
            ave_val = np.mean(val[self.nepoch - ave_last:self.nepoch], axis=0)
            if hasattr(ave_val, '__len__'):
                for i in range(len(ave_val)):
                    dict_[key + '_' + str(i)] = float(ave_val[i])
            else:
                dict_[key] = float(ave_val)
        return dict_

    def _trim_history(self, epoch):
        for key in self.history:
            self.history[key] = self.history[key][:epoch]

    @staticmethod
    def _load_pkl_data(name):
        with open(os.path.join(name, 'meta.pkl'), 'rb') as f:
            data = pickle.load(f)
            return data

    def load_model(self, name, pkl=True):
        data = {}
        if pkl:
            print('Loading', os.path.join(name, 'meta.pkl'))
            data = self._load_pkl_data(name)
        else:
            with np.load(os.path.join(name, 'meta.npz'), allow_pickle=True, encoding='latin1', fix_imports=False) as f:
                print('Loading', os.path.join(name, 'meta.npz'))
                for attr_ in f.files:
                    try:
                        data[attr_] = f[attr_][()]
                    except ModuleNotFoundError:
                        data[attr_] = None

        loss_weights = data.pop('_loss_weights')
        for attr_, val in data.items():
            if attr_ == 'config':
                self.config.update(val)
                try:  # Legacy patch
                    self.config['dense_props']['activation'] = val['dense_activation']
                except KeyError:
                    pass

            else:
                if isinstance(val, np.ndarray):
                    if val.dtype == 'O' or np.issubsctype(val.dtype, np.bytes_):
                        try:
                            val = val.astype(str)
                        except ValueError:
                            pass
                elif isinstance(val, dict):
                    pass
                elif np.issubdtype(type(val), float):
                    val = float(val)
                elif np.issubdtype(type(val), int):
                    val = int(val)
                elif np.issubdtype(type(val), np.bytes_):
                    val = val.decode('utf-8')

                # Legacy patch
                if attr_ == 'NG':
                    attr_ = 'input_dim'
                self.__setattr__(attr_, val)

        tf.keras.backend.set_floatx(self.config['floatX'])
        self._rnd = np.random.RandomState(self.seed)

        try:
            self._load_model_weights(name)
        except:
            self._build_model()
            self._load_model_weights(name, only_weights=True)
        self._init_loss_weights()
        self._set_loss_weights(loss_weights)
        self._compile_model()
        self._extract_model_parts()

    def _load_model_weights(self, name, only_weights=False):
        """ Load keras NN architecture from file `name` """

        custom_objects = dict()
        for mem_name, mem in inspect.getmembers(custom_layers):
            if inspect.isclass(mem) and issubclass(mem, tf.keras.layers.Layer):
                custom_objects[mem_name] = mem

        if only_weights:
            self.model.load_weights(os.path.join(name, 'NN.hdf5'))
        else:
            self.model = tf.keras.models.load_model(
                os.path.join(name, 'NN.hdf5'), compile=False,
                custom_objects=custom_objects
            )

    def restart(self, name, epoch):
        path = os.path.join(name, 'step%d' % epoch)
        self._load_model_weights(path)
        data = self._load_pkl_data(path)
        self._set_loss_weights(data['_loss_weights'])
        self._compile_model()

