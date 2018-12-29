from __future__ import print_function

import math
from typing import Iterable

import numpy as np


class Data:
    def __init__(self, train_data=(None, None), val_data=(None, None), test_data=(None, None), format_data=True,
                 start_dim=None, end_dim=None, to_one_hot=False, val_is_test=False):

        (self.X_train, self.Y_train) = train_data
        (self.X_val, self.Y_val) = val_data
        (self.X_test, self.Y_test) = val_data if val_is_test else test_data

        if format_data:
            if start_dim and end_dim:
                self.format_data(start_dim, end_dim, to_one_hot)
            else:
                raise Exception("Start and end dimensions must be passed to format data!")

        self.train_size = self.X_train.shape[-1] if self.X_train is not None else 0
        self.test_size = self.X_test.shape[-1] if self.X_test is not None else 0
        self.val_size = self.X_val.shape[-1] if self.X_val is not None else 0

        self.n_batches = 0
        self.start_dim = start_dim
        self.end_dim = end_dim

    def format_data(self, start_dim, end_dim, to_one_hot=False):
        self.X_train = self.format_arr(self.X_train, start_dim)
        self.X_test = self.format_arr(self.X_test, start_dim)
        self.X_val = self.format_arr(self.X_val, start_dim)

        self.Y_train = self.format_arr(self.Y_train, end_dim, to_one_hot)
        self.Y_test = self.format_arr(self.Y_test, end_dim, to_one_hot)
        self.Y_val = self.format_arr(self.Y_val, end_dim, to_one_hot)

    @staticmethod
    def format_arr(arr, dim, to_one_hot=False):
        if arr is None:
            return None

        arr = np.asarray(arr, dtype=int if to_one_hot else float)
        if arr.ndim < 2:
            arr = arr[None]  # expend dimensions

        if to_one_hot:
            return Data.convert_to_one_hot(arr, dim)

        assert dim in arr.shape
        return arr.T if arr.shape[0] != dim else arr

    @staticmethod
    def convert_to_one_hot(Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y

    @staticmethod
    def is_valid(tup):
        for x in tup:
            if x is None or not isinstance(x, Iterable):
                return False
        return True

    def mini_batches(self, batch_size=64, seed=None):
        """
        Creates a generator of random minibatches from (X_train, Y_train)

        :param batch_size: size of each mini-batch
        :param seed: seed for random shuffling the data.
        :return: generator of tuples (mini_batch_X, mini_batch_Y)
        """

        assert self.X_train is not None and self.Y_train is not None

        m = self.X_train.shape[1]  # number of training examples

        if seed is not None:
            np.random.seed(seed)

        permutation = list(np.random.permutation(m))
        shuffled_X = self.X_train[::, permutation]
        shuffled_Y = self.Y_train[::, permutation]

        self.n_batches = math.ceil(m / batch_size)
        for k in range(0, self.n_batches):
            mini_batch_X = shuffled_X[:, k * batch_size: (k + 1) * batch_size]
            mini_batch_Y = shuffled_Y[:, k * batch_size: (k + 1) * batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            yield mini_batch

    @property
    def train_set(self):
        return self.X_train, self.Y_train

    @property
    def test_set(self):
        return self.X_test, self.Y_test

    @property
    def val_set(self):
        return self.X_val, self.Y_val

    def __str__(self):
        return 'Data:\n' \
               '\tTrain: {}\n' \
               '\tTest: {}\n' \
               '\tValid: {}'.format(*[[x.shape for x in d] for d in (self.train_set, self.test_set, self.val_set)])
