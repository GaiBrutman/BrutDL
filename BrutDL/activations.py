from abc import abstractmethod

import numpy as np
from BrutDL.layer import Layer


class Activation(Layer):
    def __init__(self, func_name):
        super(Activation, self).__init__()
        self.func_name = func_name.lower()
        self.cache = None  # pre activation array from forward propagation (used for back propagation)

    @abstractmethod
    def forward(self, Z):
        self.cache = Z

    @abstractmethod
    def backward(self, dA, lr):
        pass

    def __str__(self):
        return '%s Activation Layer' % self.func_name.upper()


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__('sigmoid')

    def forward(self, Z):
        """
        Implements the SIGMOID activation in numpy

        :param Z: numpy array of any shape
        :return: output of sigmoid(z), same shape as Z
        """

        super(Sigmoid, self).forward(Z)

        A = 1 / (1 + np.exp(-Z))

        return A

    def backward(self, dA, lr):
        """
        Implement the backward propagation for a single SIGMOID unit.

        :param dA: post-activation gradient, of any shape
        :param lr: learning rate
        :return: Gradient of the cost with respect to Z
        """

        super(Sigmoid, self).backward(dA, lr)

        Z = self.cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__('softmax')

    def forward(self, Z):
        """
        Implements the SOFTMAX activation in numpy

        :param Z: numpy array of any shape
        :return: output of softmax(z), same shape as Z
        """

        super(Softmax, self).forward(Z)

        exps = np.exp(Z - np.max(Z, axis=0))
        A = exps / np.sum(exps, axis=0)

        return A

    def backward(self, dA, lr):
        """
        Implement the backward propagation for a single SOFTMAX unit.=

        :param dA: post-activation gradient, of any shape
        :param lr: learning rate
        :return: Gradient of the cost with respect to Z
        """

        super(Softmax, self).backward(dA, lr)

        Z = self.cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__('relu')

    def forward(self, Z):
        """
        Implement the RELU activation in numpy

        :param Z: Output of the linear layer, of any shape
        :return: output of relu(z), same shape as Z
        """

        super(ReLU, self).forward(Z)

        A = np.maximum(0, Z)

        return A

    def backward(self, dA, lr):
        """
        Implement the backward propagation for a single RELU unit.

        :param dA: post-activation gradient, of any shape
        :param lr: learning rate
        :return: Gradient of the cost with respect to Z
        """

        super(ReLU, self).backward(dA, lr)

        Z = self.cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z < 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ


class LeakyReLU(Activation):
    def __init__(self):
        super(LeakyReLU, self).__init__('leaky_relu')

    def forward(self, Z):
        """
        Implement the LEAKY RELU activation in numpy

        :param Z: Output of the linear layer, of any shape
        :return: output of leaky_relu(z), same shape as Z
        """

        super(LeakyReLU, self).forward(Z)

        A = np.maximum(Z, 0.01 * Z)

        return A

    def backward(self, dA, lr):
        """
        Implement the backward propagation for a single LEAKY RELU unit.

        :param dA: post-activation gradient, of any shape
        :param lr: learning rate
        :return: Gradient of the cost with respect to Z
        """

        super(LeakyReLU, self).backward(dA, lr)

        Z = self.cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        dZ[Z < 0] = 0.01

        assert (dZ.shape == Z.shape)

        return dZ


ACTIVATIONS = {'sigmoid': Sigmoid, 'softmax': Softmax, 'relu': ReLU, 'leaky_relu': LeakyReLU}
