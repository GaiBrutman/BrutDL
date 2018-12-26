import numpy as np


def softmax(Z):
    """
    Implements the SOFTMAX activation in numpy

    :param Z: numpy array of any shape
    :return: output of softmax(z), same shape as Z
    """

    exps = np.exp(Z - np.max(Z, axis=0))
    A = exps / np.sum(exps, axis=0)

    return A


def sigmoid(Z):
    """
    Implements the SIGMOID activation in numpy

    :param Z: numpy array of any shape
    :return: output of sigmoid(z), same shape as Z
    """

    A = 1 / (1 + np.exp(-Z))

    return A


def relu(Z):
    """
    Implement the RELU activation in numpy

    :param Z: Output of the linear layer, of any shape
    :return: output of relu(z), same shape as Z
    """

    A = np.maximum(0, Z)

    return A


def leaky_relu(Z):
    """
    Implement the RELU activation in numpy

    :param Z: Output of the linear layer, of any shape
    :return: output of leaky_relu(z), same shape as Z
    """

    A = np.maximum(Z, 0.01 * Z)

    return A


# ----------------------------------------------------------------------------


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    :param dA: post-activation gradient, of any shape
    :param cache: pre activation array from forward propagation
    :return: Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a single SOFTMAX unit.

    :param dA: post-activation gradient, of any shape
    :param cache: pre activation array from forward propagation
    :return: Gradient of the cost with respect to Z
    """

    dZ = sigmoid_backward(dA, cache)

    return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    :param dA: post-activation gradient, of any shape
    :param cache: pre activation array from forward propagation
    :return: Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z < 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def leaky_relu_backward(dA, cache):
    """
    Implement the backward propagation for a single LEAKY RELU unit.

    :param dA: post-activation gradient, of any shape
    :param cache: pre activation array from forward propagation
    :return: Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    dZ[Z < 0] = 0.01

    assert (dZ.shape == Z.shape)

    return dZ


# ----------------------------------------------------------------------------


def log_cost(AL, Y):
    m = Y.size

    # Compute loss from aL and y.
    c = (-1 / m) * (np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    nodes_cost = np.sum(c, axis=1)
    cost = np.squeeze(np.sum(c))

    assert cost.shape == ()

    return cost, nodes_cost


def mse_cost(AL, Y):
    # Compute loss from aL and y.
    c = 0.5 * (Y - AL) ** 2

    nodes_cost = np.mean(c, axis=1)
    cost = np.squeeze(np.mean(c))

    assert cost.shape == ()

    return cost, nodes_cost


def log_derivative(AL, Y):
    # Compute loss derivative from aL and y.
    return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))


def mse_derivative(AL, Y):
    # Compute loss derivative from aL and y.
    return -(Y - AL)


# ----------------------------------------------------------------------------


FORWARD_FUNCS = {'sigmoid': sigmoid, 'softmax': softmax, 'relu': relu, 'leaky_relu': leaky_relu}

BACKWARD_FUNCS = {'sigmoid': sigmoid_backward, 'softmax': softmax_backward, 'relu': relu_backward,
    'leaky_relu': leaky_relu_backward}

COST_FUNCS = {'mse': mse_cost, 'log': log_cost}

LOSS_DERIV_FUNCS = {'mse': mse_derivative, 'log': log_derivative}
