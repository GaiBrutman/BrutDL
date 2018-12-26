from __future__ import print_function

import numpy as np
from BrutDL.layer import WeightedLayer, Activation


class FcLayer(WeightedLayer):
    def __init__(self, out_dim, activation=None):
        super(FcLayer, self).__init__()
        
        assert isinstance(out_dim, int)

        self.in_dim = None
        self.out_dim = out_dim

        self.W = None
        self.b = None

        self.activation = Activation(activation) if activation else None

        self.cache = None

    def init_weights(self, in_dim=None):
        assert in_dim, not self.in_dim

        self.in_dim = in_dim

        self.W = (np.random.randn(self.out_dim, in_dim)) / np.sqrt(in_dim)
        self.b = np.zeros((self.out_dim, 1))

    def linear_forward(self, prev):
        Z = self.W.dot(prev) + self.b
        assert (Z.shape == (self.out_dim, prev.shape[1]))
        return Z

    def fire(self, prev):
        assert self.in_dim is not None

        Z = self.linear_forward(prev)
        A = self.activation.fire(Z) if self.activation else Z

        self.cache = (Z, prev)

        return A

    def linear_backward(self, dZ, cache):
        A_prev = cache

        m = A_prev.shape[1]

        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == self.W.shape)
        assert (db.shape == self.b.shape)

        return dA_prev, dW, db

    def back_prop(self, dA, lr):
        (Z, A) = self.cache

        dZ = self.activation.back_prop(dA, Z) if self.activation else dA

        dA_prev, dW, db = self.linear_backward(dZ, A)

        self.W = self.W - lr * dW
        self.b = self.b - lr * db

        return dA_prev

    def __str__(self):
        s = 'Fully Connected Layer: [%i -> %i]' % (self.in_dim, self.out_dim)
        if self.activation:
            s += '\n\t'+str(self.activation)
        return s