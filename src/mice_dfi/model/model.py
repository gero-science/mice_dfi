import os.path
import warnings
from tqdm.auto import trange
import numpy as np
import tensorflow as tf

from . import feeder
from . import custom_layers
from .autoencoder import BaseAutoencoder


class AE_AR(BaseAutoencoder):

    def __init__(self, *, rank=None, input_dim=None, inner_dim=None, nodes=None,
                 leaky_alpha=0.01,
                 use_bias_dense=True, dropouts=None, corrupt_input=None, bn_first=False,
                 ResNet=False, Resnet_config=None, load=None, pkl_flag=False,
                 add_ssi_raw=True, disable_bn=False, floatX='float32', **kwargs):

        base_options = dict(
            nodes=nodes,
            leaky_alpha=leaky_alpha,
            use_bias_dense=use_bias_dense,
            dropouts=dropouts,
            corrupt_input=corrupt_input,
            bn_first=bn_first,
            ResNet=ResNet,
            Resnet_config=Resnet_config,
            pkl_flag=pkl_flag,
            disable_bn=disable_bn,
            floatX=floatX,
        )
        base_options.update(kwargs)

        if load is None:
            assert rank is not None or inner_dim is not None, "Provide rank or inner dimension"
            if inner_dim is None:
                inner_dim = int(rank)

        super().__init__(inner_dim, input_dim, load, **base_options)

        # Update config
        if load is None:
            if rank is None:
                self.config['rank'] = int(inner_dim)
            else:
                self.config['rank'] = int(rank)

            self.config['add_ssi_raw'] = add_ssi_raw
            self.rank = self.config['rank']

        self._loss_weights_names = ['weight_ae', 'weight_ssi']

        # Data stratified labels
        self.labels_ssi_train = None
        self.labels_ssi_test = None

        if add_ssi_raw:
            self._loss_weights_names += ['weight_ssi_raw']

        if load is None:
            self._build_model()
            self._init_loss_weights()
            self._compile_model()
            self._extract_model_parts()

        else:
            self.load_model(load, pkl_flag)

        self._set_history_collection()
        self._sort_modes = None
        # self.rank = self.config['rank']
        self._mod_history = None  # class for mod history

    def _build_input(self):
        input_ae = super()._build_input()
        with tf.name_scope('Inputs'):
            input_ssi_x = tf.keras.layers.Input(shape=(self.input_dim,), name=self._in_ssi_x_name)
            input_ssi_y = tf.keras.layers.Input(shape=(self.input_dim,), name=self._in_ssi_y_name)
        return input_ae, input_ssi_x, input_ssi_y

    def _build_SSI(self, input_x, input_y, model=None):

        with tf.name_scope('SSI'):

            encoded_SSI_X, encoded_SSI_Y = input_x, input_y

            # allocate SSI linear dynamical part
            if model is None:
                ssi_part = custom_layers.Dense_AR(self.config['rank'], self.inner_dim, name="ssi_dynamics")
            else:
                ssi_part = model.get_layer("ssi_dynamics")

            z_SSI_X, encoded_SSI_X = ssi_part(encoded_SSI_X, is_left=True)  # tf.constant(True, dtype=tf.bool)
            z_SSI_Y, encoded_SSI_Y = ssi_part(encoded_SSI_Y, is_left=False)


            # Merge left and right SSI parts for custom loss function
            if model is None:
                merged_ssi = tf.keras.layers.Concatenate(axis=-1, name=self._out_ssi_name)(
                    [z_SSI_X, z_SSI_Y]
                )
            else:
                merged_ssi = model.get_layer(self._out_ssi_name)

            return merged_ssi, encoded_SSI_X, encoded_SSI_Y


    def _build_model(self):
        """ Build """

        # Allocate inputs
        input_ae, Data_ssi_X, Data_ssi_Y = self._build_input()
        Data_ae_X, input_ae = input_ae

        # Allocate encoder
        encoder = self._build_encoder(input_ae)

        # Wrap encoder into keras model
        model_encoder = tf.keras.models.Model(inputs=Data_ae_X, outputs=encoder, name="encoder_with_noise")  # DBG

        # SSI part
        encoded_SSI_X = self._build_encoder(Data_ssi_X, model=model_encoder, no_drops=False)
        encoded_SSI_Y = self._build_encoder(Data_ssi_Y, model=model_encoder, no_drops=False)
        merged_ssi, encoded_SSI_X, encoded_SSI_Y = self._build_SSI(encoded_SSI_X, encoded_SSI_Y)

        # Build decoder
        decoded_X = self._build_decoder(encoder)
        decoded_X = custom_layers.Identity(name=self._out_AE_name)(decoded_X)  # add name

        model_autoencoder = tf.keras.models.Model(
            inputs=Data_ae_X, outputs=decoded_X, name=self._out_AE_name,
        )

        if self.config['add_ssi_raw']:
            decoded_SSI_X = self._build_decoder(encoded_SSI_X, model=model_autoencoder)
            decoded_SSI_X = custom_layers.Identity(name=self._out_ssi_raw_name)(decoded_SSI_X)  # add name

        # Combine all parts
        if self.config['add_ssi_raw']:
            self.model = tf.keras.models.Model(
                inputs=[Data_ae_X, Data_ssi_X, Data_ssi_Y],
                outputs=[
                    decoded_X,
                    merged_ssi,
                    decoded_SSI_X,
                ])

        else:
            self.model = tf.keras.models.Model(
                inputs=[Data_ae_X, Data_ssi_X, Data_ssi_Y],
                outputs=[
                    decoded_X,
                    merged_ssi
                ])
        del model_autoencoder
        del model_encoder

    def _extract_model_parts(self):
        if self.model is None:
            raise RuntimeError("Create model first")
        super()._extract_model_parts()
        self._extract_ssi_left()
        self._extract_ssi_right()
        if self.config['add_ssi_raw']:
            self._extract_ssi_to_raw()

    def _extract_ssi_left(self):
        """ Creates estimator for Z_{n+1} from Z{n} """
        inner_ = tf.keras.layers.Input(shape=(self.inner_dim,), name='input_ssi_inner_lf')
        ssi_part = self.model.get_layer("ssi_dynamics")
        x = ssi_part(inner_, is_left=True)
        self.model_ssi_left = tf.keras.models.Model(inputs=inner_, outputs=x)

    def _extract_ssi_right(self):
        """ Creates estimator for Z_{n} """
        inner_ = tf.keras.layers.Input(shape=(self.inner_dim,), name='input_ssi_inner_rt')
        ssi_part = self.model.get_layer("ssi_dynamics")
        x = ssi_part(inner_, is_left=False)
        self.model_ssi_right = tf.keras.models.Model(inputs=inner_, outputs=x)

    def _extract_ssi_to_raw(self):

        data = tf.keras.layers.Input(shape=(self.input_dim,), name='input_ssi_raw')
        encoded_data = self._build_encoder(data, model=self.model, no_drops=True)
        ssi_part = self.model.get_layer("ssi_dynamics")
        _, encoded_data = ssi_part(encoded_data, is_left=True)
        ae_model = self.model.get_layer(self._out_AE_name)
        if not issubclass(type(ae_model), tf.keras.models.Model):
            ae_model = self.model
        decoded_data = self._build_decoder(encoded_data, model=ae_model)
        self.model_ssi_raw = tf.keras.models.Model(inputs=data, outputs=decoded_data)


    @staticmethod
    def _loss_ssi_mse(y_true, y_pred, inner, weight=1.):
        x_ssi = y_pred[:, :inner]
        y_ssi = y_pred[:, inner:]
        return tf.keras.losses.mse(y_ssi, x_ssi) * weight

    @staticmethod
    def _loss_ssi_r2(y_true, y_pred, inner, weight=1., var_mse=False):
        x_ssi = y_pred[:, :inner]
        y_ssi = y_pred[:, inner:]
        if var_mse:
            return BaseAutoencoder.tf_r2_modified(y_ssi, x_ssi) * weight
        else:
            return BaseAutoencoder.tf_r2(y_ssi, x_ssi) * weight * -1.

    @staticmethod
    def _loss_ssi_raw(y_true, y_pred, weight=1.):
        return tf.keras.losses.mse(y_true, y_pred) * weight

    def _compile_model(self, optimizer=None):
        optimizer = tf.keras.optimizers.Adam() if optimizer is None else optimizer  # Change for Adam
        weight_ae = self._loss_weights['weight_ae']
        weight_ssi = self._loss_weights['weight_ssi']
        inner_dim = self.rank
        # log_mse = self.config['log_mse']
        r2_ssi_loss = self.config['r2_ssi_loss']
        if r2_ssi_loss:
            loss_ssi_merge = AE_AR._loss_ssi_r2
        else:
            loss_ssi_merge = AE_AR._loss_ssi_mse

        if self.config['add_ssi_raw']:
            weight_ssi_raw = self._loss_weights['weight_ssi_raw']
            self.model.compile(
                optimizer=optimizer, loss={
                    self._out_AE_name: lambda x, y: AE_AR.loss_ae(x, y, weight_ae),
                    self._out_ssi_name: lambda x, y: loss_ssi_merge(x, y, inner_dim, weight_ssi),
                    self._out_ssi_raw_name: lambda x, y: AE_AR._loss_ssi_raw(x, y, weight_ssi_raw),
                }
            )
        else:
            self.model.compile(
                optimizer=optimizer, loss={
                    self._out_AE_name: lambda x, y: AE_AR.loss_ae(x, y, weight_ae),
                    self._out_ssi_name: lambda x, y: loss_ssi_merge(x, y, inner_dim, weight_ssi),
                }
            )

    def _prepare_data_ssi(self, X, Xt0, Xt1, groups=None, groups_ssi=None, labels_ssi=None, train_ratio=0.85,
                          seed_split=None, weights=None, preproc_func=None, batch=None):
        X_train, X_test = self._data_prep(
            X, groups, train_ratio, seed_split, feature_weights=weights, feature_preproc=preproc_func
        )
        X_ssi_train, X_ssi_test, ind_split = self._data_prep(
            Xt0, groups_ssi, train_ratio, seed_split,
            update_stats=False, return_index=True, index_split=None,
            stratified_labels=labels_ssi, batch=batch,
        )
        Y_ssi_train, Y_ssi_test = self._data_prep(
            Xt1, groups_ssi, train_ratio, seed_split,
            update_stats=False, return_index=False, index_split=ind_split
        )
        if labels_ssi is not None:
            assert np.isfinite(labels_ssi).all(), "Not all labels are finite"
            labels_ssi = np.array(labels_ssi, dtype=np.int32)
            labels_ssi_train = labels_ssi[ind_split[0]]
            labels_ssi_test = labels_ssi[ind_split[1]]
        else:
            labels_ssi_train, labels_ssi_test = None, None
        dummy_ssi_X = np.zeros((X_ssi_train.shape[0], self.inner_dim), dtype=self.config['floatX'])
        dummy_ssi_Y = np.zeros((X_ssi_train.shape[0], self.inner_dim), dtype=self.config['floatX'])
        dummy_ssi_X_test = np.zeros((X_ssi_test.shape[0], self.inner_dim), dtype=self.config['floatX'])
        dummy_ssi_Y_test = np.zeros((X_ssi_test.shape[0], self.inner_dim), dtype=self.config['floatX'])

        return (X_train, X_ssi_train, Y_ssi_train,), (X_test, X_ssi_test, Y_ssi_test,), \
               (X_train, dummy_ssi_X, dummy_ssi_Y,), (X_test, dummy_ssi_X_test, dummy_ssi_Y_test), \
               labels_ssi_train, labels_ssi_test

    def _init_feeders(self, batch, seed=None, future_raw=False):
        self._feeder_train = feeder.SSISequence(self.data_train, self.label_train, batch_size=batch, seed=seed,
                                                future_raw=future_raw, label_id=self.labels_ssi_train, rank=self.rank)

        # self._feeder_test = feeder.SSISequence(self.data_test, self.label_test, batch_size=batch,
        #                                        is_validation=True, seed=seed, future_raw=future_raw,
        #                                        label_id=self.labels_ssi_test)

    def train(self, X=None, Xt0=None, Xt1=None, groups=None, groups_ssi=None, labels_ssi=None, train_ratio=0.85,
              val_dts=None, nepoch=2000, batch_size=10, lr=None, seed=None, seed_split=None, resume=False,
              weight_ae=1., weight_ssi=1., weight_ssi_raw=0.,
              lambda_decay=None,
              weights_features=None, preproc_func=None, scaling='stand', restart=-1,
              plot=True, verbose=1, tbCall=False, save_dir=None, save_every=1000, debug=False, **kwargs):

        """

        Parameters
        ----------
        X
        Xt0
        Xt1
        groups
        groups_ssi
        nepoch
        batch_size
        lr
        train_ratio
        seed
        seed_split
        resume : bool, optional
            continue training from the current state. Default False
        weight_ae
        weight_ssi
        weight_every
        lambda_decay
        weights_features
        preproc_func
        scaling
        val_dts
        restart : int, optional
            continue training from the this epoch. Ignored, if < 0. Default -1. Works only with resume=True
            TBD: make implementation for complete restart (from saved dumps)
        plot
        verbose
        tbCall
        save_dir
        save_every
        debug
        """
        if restart > 0 and not resume:
            raise ValueError("If restart is enabled, than resume option should be True")

        if not resume:
            seed_split = seed if seed_split is None else seed_split
            assert (X is not None) and (Xt0 is not None) and (Xt1 is not None), "Training data is not provided"
            self.scaling = scaling
            self.data_train, self.data_test, self.label_train, \
            self.label_test, self.labels_ssi_train, self.labels_ssi_test = \
                self._prepare_data_ssi(
                    X, Xt0, Xt1, groups, groups_ssi, labels_ssi,
                    train_ratio, seed_split, weights_features, preproc_func, batch=batch_size, **kwargs
                )

            self._reset_model_weights(seed)
            self._init_feeders(batch_size, seed_split, self.config['add_ssi_raw'])

        if restart > 0:
            if save_dir is None:
                raise ValueError("Save_dir should be defined")
            self.restart(save_dir, restart)
            self._trim_history(restart)
            self.nepoch = restart

        if resume:
            if self._feeder_train is None:
                warnings.warn("Data feeder is not initialized")
                self._init_feeders(batch_size, seed_split, self.config['add_ssi_raw'])

        if val_dts is not None:
            val_dts = dict(val_dts)
            for key, val in val_dts.items():
                val_dts[key] = [self.scale(val[0].copy()), self.scale(val[1].copy())]

        # Set learning rate
        if resume and lr is not None:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        elif resume is False:
            if lr is None:
                tf.keras.backend.set_value(self.model.optimizer.lr, 0.001)  # use default value
            else:
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        # Set loss weight
        self._set_loss_weights(dict(zip(self._loss_weights_names, [weight_ae, weight_ssi, weight_ssi_raw])))

        if len(self._history_collection) > 0:
            self._set_history_args()

        callbacks = []
        tb_callback = None
        if tbCall:
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, 'tf_logs'),
                                                         write_graph=False, write_images=False, profile_batch=0)
        callbacks.append(tb_callback)

        for _ in trange(nepoch, desc="Epoch"):
            hist = self.model.fit(self._feeder_train, initial_epoch=self.nepoch,
                                  epochs=self.nepoch + 1, callbacks=callbacks, verbose=verbose)
            self._feeder_train._shuffle_data()
            self._update_history(self.nepoch)
            for key, val in hist.history.items():
                self._append_to_dict(self.history, key, val[0], self.nepoch)
            self.nepoch += 1
            self._save_every(save_dir, save_every)

    def predict_decoded(self, X, scale=True):
        dX = self.model_autoencoder.predict(self.scale(X, scale=scale))
        return self.unscale(dX, scale=scale)

    def predict_latent(self, X, scale=True):
        """ """
        return self.model_encoder.predict(self.scale(X, scale=scale))

    def predict_next_x(self, X, scale=True):
        pred_x = self.model_ssi_raw.predict(self.scale(X, scale=scale))
        return self.unscale(pred_x, scale=scale)

    def predict_inner(self, X, scale=True, is_left=False, project_b=False):

        inner = self.model_encoder.predict(self.scale(X, scale=scale))

        if is_left:
            res = self.model_ssi_left.predict(inner)
        else:
            res = self.model_ssi_right.predict(inner)

        if project_b:
            return res[1]
        else:
            return res[0]

    def predict_left_z(self, X, scale=True):
        return self.predict_inner(X, scale, is_left=True, project_b=False)

    def predict_left_zb(self, X, scale=True):
        return self.predict_inner(X, scale, is_left=True, project_b=True)

    def predict_right_z(self, X, scale=True):
        """ Predict Z-scores (modes) """
        return self.predict_inner(X, scale, is_left=False, project_b=False)

    def predict_right_zb(self, X, scale=True):
        """ Predict Z-scores (modes) """
        return self.predict_inner(X, scale, is_left=False, project_b=True)

    def predict_z(self, X, scale=True):
        """ Predict Z-scores (modes) """
        return self.predict_right_z(X, scale)

    def _predict_x_y_ssi(self, x_true, y_true, scale, latent, **kwargs):

        if latent:
            x_pred = self.predict_left_zb(x_true, scale=scale)
            y_pred = self.predict_right_zb(y_true, scale=scale)
        else:
            x_pred = self.predict_left_z(x_true, scale=scale)
            y_pred = self.predict_right_z(y_true, scale=scale)
        return x_pred, y_pred

    def _set_eigs(self, sorting=None):
        ssi_layer = self.model.get_layer('ssi_dynamics')  # .get_weights()
        eigs = tf.keras.backend.get_value(ssi_layer.eigvals)
        self._va = tf.keras.backend.get_value(ssi_layer.ssi_A)
        self._vb = tf.keras.backend.get_value(ssi_layer.ssi_B)
        self._shift = tf.keras.backend.get_value(ssi_layer.shift_z)

        # Sort from max->min
        if sorting is None:
            sort_eig = np.argsort(eigs)[::-1]
            self.eigs = eigs[sort_eig]
            self._sort_modes = sort_eig
        else:
            assert len(sorting) == self.rank
            assert len(sorting) == len(np.unique(sorting))
            self._sort_modes = sorting

        self._vb = self._vb[self._sort_modes, :]
        self._va = self._va[:, self._sort_modes]

    def get_lambdas(self):
        self._set_eigs()
        return self.eigs

    def get_constraint_ssi(self):
        constraint = self.model.get_layer('ssi_dynamics').constraint_matrices()
        return tf.keras.backend.get_value(constraint)

    def get_eigs_weights(self):
        self._set_eigs()
        return self.eigs, self._va, self._vb

    def get_eigs(self):
        self._set_eigs()
        return self.eigs

    def get_ssi_loss(self, x_true, y_true, scale=True, loss='mse', latent=False):
        x_pred, y_pred = self._predict_x_y_ssi(x_true, y_true, scale, latent)
        if loss == 'mse':
            return self.loss_mse(x_pred, y_pred)
        elif loss == 'lmse':
            return self.loss_lmse(x_pred, y_pred)

    def get_ssi_score(self, x_true, y_true, scale=True, latent=False):
        x_pred, y_pred = self._predict_x_y_ssi(x_true, y_true, scale, latent)
        return self.score_r2(y_pred, x_pred)

    def get_ssi_raw_loss(self, x_true, y_true, scale=True, loss='mse'):
        y_pred = self.predict_next_x(x_true, scale)
        if loss == 'mse':
            return self.loss_mse(y_true, y_pred)
        elif loss == 'lmse':
            return self.loss_lmse(y_true, y_pred)

    def get_ssi_raw_score(self, x_true, y_true, scale=True):
        y_pred = self.predict_next_x(x_true, scale)
        return self.score_r2(y_true, y_pred)

    def get_ssi_corr(self, x_true, y_true, scale=True):
        x_pred, y_pred = self._predict_x_y_ssi(x_true, y_true, scale, latent=False)
        return self.correlation(y_pred, x_pred)

    def get_mean_var(self, data, inner=True, var=False):
        eigs, a, b = self.get_eigs_weights()
        latent_x = self.predict_latent(data, scale=False)
        if not inner:
            latent_x = np.dot(latent_x, a)
        if var:
            return np.var(latent_x, axis=0)
        else:
            return np.mean(latent_x, axis=0)

    def _set_history_collection(self):
        super()._set_history_collection()
        hist = self._history_collection
        hist['loss_tr_tf'] = 'get_default_loss'
        hist['loss_te_tf'] = 'get_default_loss'
        hist['r2_tr_ae'] = 'get_ae_score'
        hist['r2_te_ae'] = 'get_ae_score'
        hist['loss_tr_lat_ssi'] = 'get_ssi_loss'
        hist['loss_te_lat_ssi'] = 'get_ssi_loss'
        hist['r2_tr_lat_ssi'] = 'get_ssi_score'
        hist['r2_te_lat_ssi'] = 'get_ssi_score'
        hist['loss_tr_ssi'] = 'get_ssi_loss'
        hist['loss_te_ssi'] = 'get_ssi_loss'
        hist['r2_tr_tot_ssi'] = 'get_ssi_score'
        hist['r2_te_tot_ssi'] = 'get_ssi_score'
        hist['r2_tr_ssi'] = 'get_ssi_corr'
        hist['r2_te_ssi'] = 'get_ssi_corr'
        hist['lr'] = 'get_model_var'
        hist['w_ae'] = 'get_model_var'
        hist['w_ssi'] = 'get_model_var'
        hist['eigs'] = 'get_eigs'
        hist['constrain'] = 'get_constraint_ssi'
        hist['mean_bot'] = 'get_mean_var'
        hist['var_bot'] = 'get_mean_var'

        if self.config['add_ssi_raw']:
            hist['w_ssi_raw'] = 'get_model_var'
            hist['loss_tr_raw_ssi'] = 'get_ssi_raw_loss'
            hist['loss_te_raw_ssi'] = 'get_ssi_raw_loss'
            hist['r2_tr_raw_ssi'] = 'get_ssi_raw_score'
            hist['r2_te_raw_ssi'] = 'get_ssi_raw_score'

        self._history_collection = hist

    def _set_history_args(self):
        train_feed = feeder.SSISequence(self.data_train, self.label_train, batch_size=None,
                                        is_validation=True, future_raw=self.config['add_ssi_raw'])
        test_feed = feeder.SSISequence(self.data_test, self.label_test, batch_size=None,
                                       is_validation=True, future_raw=self.config['add_ssi_raw'])
        args = self._history_args
        args['loss_tr_tf'] = (train_feed, 1)
        args['loss_te_tf'] = (test_feed, 1)
        args['r2_tr_ae'] = (self.data_train[0], False)
        args['r2_te_ae'] = (self.data_test[0], False)
        args['loss_tr_lat_ssi'] = (self.data_train[1], self.data_train[2], False, 'mse', True)
        args['loss_te_lat_ssi'] = (self.data_test[1], self.data_test[2], False, 'mse', True)
        args['r2_tr_lat_ssi'] = (self.data_train[1], self.data_train[2], False, True)  # 'get_ssi_score'
        args['r2_te_lat_ssi'] = (self.data_test[1], self.data_test[2], False, True)
        args['loss_tr_ssi'] = (self.data_train[1], self.data_train[2], False, 'mse', False)
        args['loss_te_ssi'] = (self.data_test[1], self.data_test[2], False, 'mse', False)
        args['r2_tr_tot_ssi'] = (self.data_train[1], self.data_train[2], False, False)
        args['r2_te_tot_ssi'] = (self.data_test[1], self.data_test[2], False, False)
        args['r2_tr_ssi'] = (self.data_train[1], self.data_train[2], False)  # 'get_ssi_corr'
        args['r2_te_ssi'] = (self.data_test[1], self.data_test[2], False)  # 'get_ssi_corr'
        args['lr'] = (self.model.optimizer.lr,)
        args['w_ae'] = (self._loss_weights['weight_ae'],)
        args['w_ssi'] = (self._loss_weights['weight_ssi'],)
        args['eigs'] = ()
        args['constrain'] = ()
        args['mean_bot'] = (self.data_train[1], True, False)
        args['var_bot'] = (self.data_train[1], True, True)

        if self.config['add_ssi_raw']:
            args['w_ssi_raw'] = (self._loss_weights['weight_ssi_raw'],)
            args['loss_tr_raw_ssi'] = (self.data_train[1], self.data_train[2], False, 'mse')
            args['loss_te_raw_ssi'] = (self.data_test[1], self.data_test[2], False, 'mse')
            args['r2_tr_raw_ssi'] = (self.data_train[1], self.data_train[2], False)
            args['r2_te_raw_ssi'] = (self.data_test[1], self.data_test[2], False)

        self._history_args = args


class AE_AR_sex(AE_AR):
    def __init__(self, rank=None, input_dim=None, inner_dim=None, nodes=None, dense_activation='linear', leaky_alpha=0,
                 use_bias_dense=True, dropouts=None, corrupt_input=None, bn_first=False,
                 ResNet=False, Resnet_config=None, load=None, pkl_flag=True,
                 add_ssi_raw=True, disable_bn=False, floatX='float32', **kwargs):

        base_options = dict(
            rank=rank,
            inner_dim=inner_dim,
            input_dim=input_dim,
            load=load,
            nodes=nodes,
            dense_activation=dense_activation,
            leaky_alpha=leaky_alpha,
            use_bias_dense=use_bias_dense,
            dropouts=dropouts,
            corrupt_input=corrupt_input,
            bn_first=bn_first,
            ResNet=ResNet,
            Resnet_config=Resnet_config,
            pkl_flag=pkl_flag,
            add_ssi_raw=add_ssi_raw,
            disable_bn=disable_bn,
            floatX=floatX,
        )
        base_options.update(kwargs)
        super().__init__(**base_options)

    def _build_model(self):
        """ Build """

        # Allocate inputs
        input_ae, Data_ssi_X, Data_ssi_Y, Data_sex = self._build_input()
        Data_ae_X, input_ae = input_ae

        # Allocate encoder
        encoder = self._build_encoder(input_ae)

        # Wrap encoder into keras model
        model_encoder = tf.keras.models.Model(inputs=Data_ae_X, outputs=encoder, name="encoder_with_noise")  # DBG

        # SSI part
        encoded_SSI_X = self._build_encoder(Data_ssi_X, model=model_encoder, no_drops=False)
        encoded_SSI_Y = self._build_encoder(Data_ssi_Y, model=model_encoder, no_drops=False)
        merged_ssi, encoded_SSI_X, encoded_SSI_Y = self._build_SSI(encoded_SSI_X, encoded_SSI_Y, Data_sex)

        # Build decoder
        decoded_X = self._build_decoder(encoder)
        decoded_X = custom_layers.Identity(name=self._out_AE_name)(decoded_X)  # add name

        model_autoencoder = tf.keras.models.Model(
            inputs=Data_ae_X, outputs=decoded_X, name=self._out_AE_name,
        )

        if self.config['add_ssi_raw']:
            decoded_SSI_X = self._build_decoder(encoded_SSI_X, model=model_autoencoder)
            decoded_SSI_X = custom_layers.Identity(name=self._out_ssi_raw_name)(decoded_SSI_X)  # add name

        # Combine all parts
        if self.config['add_ssi_raw']:
            self.model = tf.keras.models.Model(
                inputs=[Data_ae_X, Data_ssi_X, Data_ssi_Y, Data_sex],
                outputs=[
                    decoded_X,
                    merged_ssi,
                    decoded_SSI_X,
                ])

        else:
            self.model = tf.keras.models.Model(
                inputs=[Data_ae_X, Data_ssi_X, Data_ssi_Y, Data_sex],
                outputs=[
                    decoded_X,
                    merged_ssi
                ])
        del model_autoencoder
        del model_encoder

    def _build_input(self):
        input_ae, input_ssi_x, input_ssi_y = super()._build_input()
        with tf.name_scope('Inputs'):
            input_sex = tf.keras.layers.Input(shape=(1,), name='Input_sex')
        return input_ae, input_ssi_x, input_ssi_y, input_sex

    def _build_SSI(self, input_x, input_y, input_t, model=None):

        with tf.name_scope('SSI'):
            encoded_SSI_X, encoded_SSI_Y = input_x, input_y

            if model is None:
                ssi_part = custom_layers.Dense_AR_sex(self.config['rank'], self.inner_dim,
                                                      name="ssi_dynamics")  # Dense_SSI_cond
            else:
                ssi_part = model.get_layer("ssi_dynamics")

            z_SSI_X, encoded_SSI_X = ssi_part([encoded_SSI_X, input_t],
                                              is_left=True)  # tf.constant(True, dtype=tf.bool)
            z_SSI_Y, encoded_SSI_Y = ssi_part([encoded_SSI_Y, input_t], is_left=False)

            # Merge left and right SSI parts for custom loss function
            if model is None:
                merged_ssi = tf.keras.layers.Concatenate(axis=-1, name=self._out_ssi_name)(
                    [z_SSI_X, z_SSI_Y]
                )
            else:
                merged_ssi = model.get_layer(self._out_ssi_name)

            return merged_ssi, encoded_SSI_X, encoded_SSI_Y

    def _extract_ssi_to_raw(self):
        if self.model is None:
            raise RuntimeError("Create model first")
        data = tf.keras.layers.Input(shape=(self.input_dim,), name='input_ssi_raw')
        sex = tf.keras.layers.Input(shape=(1,), name='input_sex')

        encoded_data = self._build_encoder(data, model=self.model, no_drops=True)
        ssi_part = self.model.get_layer("ssi_dynamics")
        _, encoded_data = ssi_part([encoded_data, sex], is_left=True)
        ae_model = self.model.get_layer(self._out_AE_name)
        if not issubclass(type(ae_model), tf.keras.models.Model):
            ae_model = self.model
        decoded_data = self._build_decoder(encoded_data, model=ae_model)
        self.model_ssi_raw = tf.keras.models.Model(inputs=[data, sex], outputs=decoded_data)

    def _extract_ssi_left(self):
        """ Creates estimator for Z_{n+1} from Z{n} """
        inner_ = tf.keras.layers.Input(shape=(self.inner_dim,), name='input_ssi_inner_lf')
        sex_ = tf.keras.layers.Input(shape=(1,), name='input_ssi_sex_lf')

        ssi_part = self.model.get_layer("ssi_dynamics")
        x = ssi_part([inner_, sex_], is_left=True)
        self.model_ssi_left = tf.keras.models.Model(inputs=[inner_, sex_], outputs=x)

    def _extract_ssi_right(self):
        """ Creates estimator for Z_{n} """
        inner_ = tf.keras.layers.Input(shape=(self.inner_dim,), name='input_ssi_inner_rt')
        sex_ = tf.keras.layers.Input(shape=(1,), name='input_ssi_sex_rt')

        ssi_part = self.model.get_layer("ssi_dynamics")
        x = ssi_part([inner_, sex_], is_left=False)
        self.model_ssi_right = tf.keras.models.Model(inputs=[inner_, sex_], outputs=x)

    def _init_feeders(self, batch, seed=None, future_raw=False):
        self._feeder_train = feeder.SSISequence_sex(self.data_train, self.label_train, batch_size=batch, seed=seed,
                                                    future_raw=future_raw, label_id=self.labels_ssi_train)
        # self._feeder_test = feeder.SSISequence(self.data_test, self.label_test, batch_size=batch,
        #                                        is_validation=True, seed=seed, future_raw=future_raw,
        #                                        label_id=self.labels_ssi_test)

    def _prepare_data_ssi(self, X, Xt0, Xt1, groups=None, groups_ssi=None, labels_ssi=None, train_ratio=0.85,
                          seed_split=None, weights=None, preproc_func=None, batch=None, **kwargs):
        X_train, X_test = self._data_prep(
            X, groups, train_ratio, seed_split, feature_weights=weights, feature_preproc=preproc_func
        )
        X_ssi_train, X_ssi_test, ind_split = self._data_prep(
            Xt0, groups_ssi, train_ratio, seed_split,
            update_stats=False, return_index=True, index_split=None,
            stratified_labels=labels_ssi, batch=batch,
        )
        Y_ssi_train, Y_ssi_test = self._data_prep(
            Xt1, groups_ssi, train_ratio, seed_split,
            update_stats=False, return_index=False, index_split=ind_split
        )

        if labels_ssi is None:
            raise ValueError("Parameter labels_ssi should be provided when running autoencoder with time lag")

        assert np.isfinite(labels_ssi).all(), "Not all labels are finite"
        labels_ssi = np.array(labels_ssi, dtype=np.int32)
        labels_ssi_train = labels_ssi[ind_split[0]]
        labels_ssi_test = labels_ssi[ind_split[1]]

        sex = kwargs.get('sex', None)
        if sex is None:
            raise ValueError('Sex is not provided')
        sex = sex.astype(self.config['floatX'])
        check = np.isfinite(sex) & (sex >= 0)
        assert check.sum() == len(check), "Timelag array contains zeros of NaNs"

        sex_train = sex[ind_split[0]]
        sex_test = sex[ind_split[1]]

        dummy_ssi_X = np.zeros((X_ssi_train.shape[0], self.inner_dim), dtype=self.config['floatX'])
        dummy_ssi_Y = np.zeros((X_ssi_train.shape[0], self.inner_dim), dtype=self.config['floatX'])
        dummy_ssi_X_test = np.zeros((X_ssi_test.shape[0], self.inner_dim), dtype=self.config['floatX'])
        dummy_ssi_Y_test = np.zeros((X_ssi_test.shape[0], self.inner_dim), dtype=self.config['floatX'])

        return (X_train, X_ssi_train, Y_ssi_train, sex_train), (X_test, X_ssi_test, Y_ssi_test, sex_test), \
               (X_train, dummy_ssi_X, dummy_ssi_Y,), (X_test, dummy_ssi_X_test, dummy_ssi_Y_test), \
               labels_ssi_train, labels_ssi_test

    def predict_inner(self, X, scale=True, is_left=False, project_b=False):

        data, sex = X
        inner = self.model_encoder.predict(self.scale(data, scale=scale))

        if is_left:
            res = self.model_ssi_left.predict([inner, sex])
        else:
            res = self.model_ssi_right.predict([inner, sex])

        if project_b:
            return res[1]
        else:
            return res[0]

    def predict_next_x(self, X, scale=True):
        data_x, sex = X
        pred_x = self.model_ssi_raw.predict([self.scale(data_x, scale=scale), sex])
        return self.unscale(pred_x, scale=scale)

    def predict_z(self, X, scale=True):
        """ Predict Z-scores (modes) """
        dummy = np.zeros(X.shape[0])
        return self.predict_right_z([X, dummy], scale)

    def get_lambdas(self):
        self._set_eigs()
        return self.eigs

    def _predict_x_y_ssi(self, x_true, y_true, scale, latent, **kwargs):
        sex = kwargs.pop('sex')

        if latent:
            x_pred = self.predict_left_zb([x_true, sex], scale=scale)
            y_pred = self.predict_right_zb([y_true, sex], scale=scale)
        else:
            x_pred = self.predict_left_z([x_true, sex], scale=scale)
            y_pred = self.predict_right_z([y_true, sex], scale=scale)
        return x_pred, y_pred

    def get_ssi_loss(self, x_true, y_true, sex, scale=True, loss='mse', latent=False):
        x_pred, y_pred = self._predict_x_y_ssi(x_true, y_true, scale, latent, sex=sex)
        if loss == 'mse':
            return self.loss_mse(x_pred, y_pred)
        elif loss == 'lmse':
            return self.loss_lmse(x_pred, y_pred)

    def get_ssi_score(self, x_true, y_true, sex, scale=True, latent=False):
        x_pred, y_pred = self._predict_x_y_ssi(x_true, y_true, scale, latent, sex=sex)
        return self.score_r2(y_pred, x_pred)

    def get_ssi_raw_loss(self, x_true, y_true, scale=True, loss='mse'):
        y_pred = self.predict_next_x(x_true, scale)
        if loss == 'mse':
            return self.loss_mse(y_true, y_pred)
        elif loss == 'lmse':
            return self.loss_lmse(y_true, y_pred)

    def get_ssi_raw_score(self, x_true, y_true, scale=True):
        y_pred = self.predict_next_x(x_true, scale)
        return self.score_r2(y_true, y_pred)

    def get_ssi_corr(self, x_true, y_true, sex, scale=True):
        x_pred, y_pred = self._predict_x_y_ssi(x_true, y_true, scale, False, sex=sex)
        return self.correlation(y_pred, x_pred)

    def _set_history_args(self):
        train_feed = feeder.SSISequence_sex(
            self.data_train, self.label_train, batch_size=None,
            is_validation=True, add_ssi_raw=self.config['add_ssi_raw'],
            label_id=self.labels_ssi_train)
        test_feed = feeder.SSISequence_sex(
            self.data_test, self.label_test, batch_size=None,
            is_validation=True, add_ssi_raw=self.config['add_ssi_raw'],
            label_id=self.labels_ssi_test)

        args = self._history_args
        args['loss_tr_tf'] = (train_feed, 1)
        args['loss_te_tf'] = (test_feed, 1)
        args['r2_tr_ae'] = (self.data_train[0], False)
        args['r2_te_ae'] = (self.data_test[0], False)
        args['loss_tr_lat_ssi'] = (self.data_train[1], self.data_train[2], self.data_train[3], False, 'mse', True)
        args['loss_te_lat_ssi'] = (self.data_test[1], self.data_test[2], self.data_test[3], False, 'mse', True)
        args['r2_tr_lat_ssi'] = (
        self.data_train[1], self.data_train[2], self.data_train[3], False, True)  # 'get_ssi_score'
        args['r2_te_lat_ssi'] = (self.data_test[1], self.data_test[2], self.data_test[3], False, True,)
        args['loss_tr_ssi'] = (self.data_train[1], self.data_train[2], self.data_train[3], False, 'mse', False,)
        args['loss_te_ssi'] = (self.data_test[1], self.data_test[2], self.data_test[3], False, 'mse', False)
        args['r2_tr_tot_ssi'] = (self.data_train[1], self.data_train[2], self.data_train[3], False, False)
        args['r2_te_tot_ssi'] = (self.data_test[1], self.data_test[2], self.data_test[3], False, False)
        args['r2_tr_ssi'] = (self.data_train[1], self.data_train[2], self.data_train[3], False)  # 'get_ssi_corr'
        args['r2_te_ssi'] = (self.data_test[1], self.data_test[2], self.data_test[3], False)  # 'get_ssi_corr'
        args['lr'] = (self.model.optimizer.lr,)
        args['w_ae'] = (self._loss_weights['weight_ae'],)
        args['w_ssi'] = (self._loss_weights['weight_ssi'],)
        args['eigs'] = ()
        args['constrain'] = ()
        args['mean_bot'] = (self.data_train[1], True, False)
        args['var_bot'] = (self.data_train[1], True, True)
        if self.config['add_ssi_raw']:
            args['w_ssi_raw'] = (self._loss_weights['weight_ssi_raw'],)
            args['loss_tr_raw_ssi'] = ((self.data_train[1], self.data_train[3],), self.data_train[2], False, 'mse')
            args['loss_te_raw_ssi'] = ((self.data_test[1], self.data_test[3],), self.data_test[2], False, 'mse')
            args['r2_tr_raw_ssi'] = ((self.data_train[1], self.data_train[3],), self.data_train[2], False)
            args['r2_te_raw_ssi'] = ((self.data_test[1], self.data_test[3],), self.data_test[2], False)

        self._history_args = args
