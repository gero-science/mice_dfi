# -*- coding: utf8 -*-
"""
Model feeders
"""

import numpy as np
import tensorflow as tf


class SSISequence(tf.keras.utils.Sequence):

    def __init__(self, inputs, labels, batch_size, label_id=None, is_validation=False, seed=None, future_raw=False,
                 cut='round', rank=None):

        # Allocate data
        self.data_AE = np.empty(0)
        self.data_x_SSI = np.empty(0)
        self.data_y_SSI = np.empty(0)
        self._init_data(inputs, labels)

        self.min_length = min(len(self.data_AE), len(self.data_x_SSI))

        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = self.min_length

        floatx = self.data_AE.dtype
        if rank is None:
            rank = self.data_x_SSI.shape[1]
        self.dummy_SSI = np.zeros((self.batch_size, rank*2), dtype=floatx)
        self.validation = is_validation
        self.rnd = np.random.RandomState(seed)
        self.future_raw = future_raw
        self.ind_ae = np.arange(len(self.data_AE), dtype=np.int32)
        self.ind_ssi = np.arange(len(self.data_x_SSI), dtype=np.int32)
        self.min_length = min(len(self.data_AE), len(self.data_x_SSI))

        self.labels = None
        if label_id is not None and is_validation is False:
            self._init_cut_strategy(cut)
            self.labels = np.array(label_id, dtype=np.int32)
            assert len(self.data_x_SSI) == len(label_id), "Size of labels %d doesn't match size of data %d" % (
                len(self.data_x_SSI), len(label_id))
            self._uvals, self._ucounts = np.unique(self.labels, return_counts=True, axis=0)
            if (self._ucounts < self.batch_size).any():
                m = self._ucounts < self.batch_size
                raise ValueError("List of %s items in labels have size %s smaller than batch size %d" % (
                    self._uvals[m], self._ucounts[m], self.batch_size))
            # increase size of SSI
            ssi_length = int(batch_size * self.round(self._ucounts * 1. / batch_size).sum())
            self.min_length = min(len(self.data_AE), ssi_length)

        self._shuffle_data()

    def _init_data(self, inputs, labels):
        self.data_AE, self.data_x_SSI, self.data_y_SSI = inputs
        self.data_AE_true, _, _ = labels

    def __len__(self):
        return int(np.floor(self.min_length / self.batch_size))

    def __getitem__(self, idx):
        pos1 = idx * self.batch_size
        pos2 = pos1 + self.batch_size
        batch_x_ae = self.data_AE[self.ind_ae][pos1:pos2]
        batch_x_ssi = self.data_x_SSI[self.ind_ssi][pos1:pos2]
        batch_y_ssi = self.data_y_SSI[self.ind_ssi][pos1:pos2]
        if self.future_raw:
            return [batch_x_ae, batch_x_ssi, batch_y_ssi], [batch_x_ae, self.dummy_SSI, batch_y_ssi]
        else:
            return [batch_x_ae, batch_x_ssi, batch_y_ssi], [batch_x_ae, self.dummy_SSI]

    def _shuffle_data(self):
        if not self.validation:
            self.rnd.shuffle(self.ind_ae)
            self.rnd.shuffle(self.ind_ssi)

        if self.labels is not None and self.validation is False:
            self.shuffle_stratified_labels()

    def _init_cut_strategy(self, cut):
        assert cut in ['ceil', 'floor', 'round'], "Unknown cut value"
        if cut == 'ceil':
            self.round = np.ceil
        elif cut == 'floor':
            self.round = np.floor
        elif cut == 'round':
            self.round = np.round

    def shuffle_stratified_labels(self):
        labels = self.labels[self.ind_ssi]
        uvals, ucounts = self._uvals, self._ucounts
        batch_alignment = np.empty((0, self.batch_size), dtype=np.int32)
        for i in range(len(ucounts)):
            ind = np.equal(labels, uvals[i])
            if labels.ndim > 1:
                ind = ind.all(axis=1)
            ind = np.argwhere(ind)
            complete = len(ind) // self.batch_size  # completed batches
            target = int(
                self.round(len(ind) * 1. / self.batch_size))  # desired number of batches according rounding strategy
            additional = 0
            if target > complete:
                additional = self.batch_size - len(ind) % self.batch_size
            if additional > 0:
                ind = np.append(ind,
                                self.rnd.choice(ind[:complete * self.batch_size].flatten(), additional, replace=False))
            else:
                ind = self.rnd.choice(ind.flatten(), size=complete * self.batch_size, replace=False)

            batch_alignment = np.concatenate((batch_alignment, ind.reshape(-1, self.batch_size)))

        self.rnd.shuffle(batch_alignment)
        batch_alignment = batch_alignment.flatten()
        self.ind_ssi = self.ind_ssi[batch_alignment]

    def on_epoch_end(self):
        self._shuffle_data()


class SSISequence_sex(SSISequence):

    def _init_data(self, inputs, labels):
        self.data_AE, self.data_x_SSI, self.data_y_SSI, self.data_sex = inputs
        self.data_AE_true, _, _ = labels

    def __getitem__(self, idx):
        pos1 = idx * self.batch_size
        pos2 = pos1 + self.batch_size
        batch_x_ae = self.data_AE[self.ind_ae][pos1:pos2]
        batch_x_ssi = self.data_x_SSI[self.ind_ssi][pos1:pos2]
        batch_y_ssi = self.data_y_SSI[self.ind_ssi][pos1:pos2]
        batch_sex = self.data_sex[self.ind_ssi][pos1:pos2]
        if self.future_raw:
            return [batch_x_ae, batch_x_ssi, batch_y_ssi, batch_sex], [batch_x_ae, self.dummy_SSI, batch_y_ssi]
        else:
            return [batch_x_ae, batch_x_ssi, batch_y_ssi, batch_sex], [batch_x_ae, self.dummy_SSI]
