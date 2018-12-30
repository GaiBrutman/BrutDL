from __future__ import print_function

import numpy as np

import math

import matplotlib.pyplot as plt

from BrutDL.data import Data
from BrutDL.costs import Cost, COSTS
from BrutDL.layer import Layer, WeightedLayer
from BrutDL.print_utils import PrintUtils


class Model:
    def __init__(self, input_dim, layers=None, cost='mse'):
        """
        Initialises of the class

        :param input_dim: the shape of the input data
        :param layers: list of model layers
        :param cost: the cost to use in back propagation
        """

        super(Model, self).__init__()

        self.__layers_dims = [input_dim]  # List of array dimensions (int) in the model

        self.__layers = []

        self.__activations = []  # List of activation (str) functions applied in the model

        if isinstance(cost, str):
            assert cost in COSTS
            cost = COSTS[cost]()

        assert isinstance(cost, Cost)
        self.__cost = cost  # Cost object

        self.__costs = []  # Keep track of cost (list of floats)

        self.__last_batch_size = 0  # Last number of training samples is each mini bach
        self.__last_train_size = 0  # Last number of total training samples
        self.__last_lr = 0.  # Last training learning rate.

        self.__accuracies = {'Train': 0., 'Test': 0.}

        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """
        Adds a layer to the model

        :param layer: the deep learning layer to add
        :return: None
        """

        assert isinstance(layer, Layer)

        self.__layers.append(layer)

        if isinstance(layer, WeightedLayer):
            assert self.__layers_dims
            layer.init_weights(self.__layers_dims[-1])
            self.__layers_dims.append(layer.out_dim)
            if layer.activation:
                self.__activations.append(str(layer.activation.func_name))

    def forward(self, X):
        """
        Runs forward propagation

        :param X: data to be passed, numpy array of shape (input size, number of examples)
        :return: last post-activation value

        """

        A = X

        # run current layer forward, Add "cache" to the "caches" list.
        for l in self.__layers:
            A = l.forward(A)

        return A

    def backward(self, AL, Y, lr):
        """
        Runs backward propagation, updates parameters for all layers

        :param AL: last post-activation value (from forward propagation)
        :param Y: data correct labels
        :param lr: the backward propagation learning rate
        :return: None
        """

        dAL = self.__cost.derivative(AL, Y)

        dA = dAL
        for l in reversed(self.__layers):
            dA = l.backward(dA, lr)

    def train(self, data=None, train_data=(None, None), val_data=(None, None), lr=1e-3, epochs=7, batch_size=64,
              verbose=True, lr_descent=False, seed=None):
        """
        Trains the network.

        :param data: 'Data' object contains train and (optional) validation data
        :param train_data: tuple containing X and Y arrays of train data (replaces the 'data' param)
        :param val_data: tuple containing X and Y arrays of validation data (replaces the 'data' param)
        :param lr: training learning rate
        :param epochs: number of train epochs
        :param batch_size: size of each mini-batch of data
        :param verbose: whether to print the training process
        :param lr_descent: whether to descent the learning rate throughout each epoch
        :param seed: seed state to initialize the numpy.random generator
        :return: None
        """

        np.random.seed(seed)

        assert data is not None or Data.is_valid(train_data)

        if data is None:
            data = Data(train_data=train_data, val_data=val_data, start_dim=self.__layers_dims[0],
                        end_dim=self.__layers_dims[-1])

        assert isinstance(data, Data)

        self.__last_train_size = data.train_size
        self.__last_batch_size = batch_size
        self.__last_lr = lr

        # Loop over epochs
        for epoch in range(epochs):
            lr = self.__last_lr

            batch_cost = np.zeros(data.end_dim)
            # Loop over epochs
            for j, (min_batch_x, min_batch_y) in enumerate(data.mini_batches(batch_size, seed=seed)):
                # Forward propagation
                AL = self.forward(min_batch_x)

                # Compute cost
                cost, nodes_cost = self.__cost.compute(AL, min_batch_y, get_node_cost=True)
                batch_cost += nodes_cost / data.n_batches

                # Back propagation
                self.backward(AL, min_batch_y, lr)

                # Print the training progress
                if verbose:
                    PrintUtils.print_progress(j + 1, data.n_batches, epoch, cost,
                                              'Lr: %f' % lr if lr_descent else '')

                if lr_descent:
                    lr = (self.__last_lr / 2.) * math.cos((j / data.n_batches) * math.pi) + (self.__last_lr / 2.)

            self.__costs.append(batch_cost)

            # Print progress of each epoch
            if verbose:
                # If validation data exists, print validation cost
                if Data.is_valid(data.val_set):
                    val_cost = self.__cost.compute(self.forward(data.X_val), data.Y_val)
                else:
                    val_cost = None

                PrintUtils.print_progress(j + 1, data.n_batches, epoch, np.mean(batch_cost),
                                          'Lr: %f' % lr if lr_descent else '', 'Val Cost: %f' % val_cost
                                          if val_cost is not None else '')

                print()

        print('\nDONE\n')

    def predict(self, X):
        """
        Wrapper for forward() function, used to get post train predictions.
        :param X: data to be passed, numpy array of shape (input size, number of examples)
        :return: the model's prediction - return value of forward()
        """

        pred = self.forward(Data.format_arr(X, self.__layers_dims[0]))
        return pred

    def classifier_accuracy(self, dataset, type_='Test', verbose=True):
        """
        Calculates the accuracy of the model on given data.

        :param dataset: tuple of Data to be passed and labels for calculating the accuracy
        :param type_: type of data given. Used for saving the accuracy value in the right spot
        :param verbose: whether to print the accuracy
        :return: the accuracy value of the model on the given data.
        """

        assert type_ in self.__accuracies

        (X, Y) = dataset

        p = np.argmax(self.predict(X), axis=0)
        r = np.argmax(Y, axis=0)

        acc = float(np.mean(p == r))
        self.__accuracies[type_] = acc

        if verbose:
            print('pred:', p[:10])
            print('true:', r[:10])

            print('{} Accuracy: {}\n'.format(type_, acc))

        return acc

    def plot_cost(self):
        """
        Plots the cost history of the model, also shows different model parameters and accuracies.
        :return: the plot figure
        """

        title = 'Layers: {}\n' \
                'Activations: {}\n\n' \
                'Size = {}, Batch Size = {}, Learning Rate = {}'.format(self.__layers_dims, self.__activations,
                                                                        self.__last_train_size, self.__last_batch_size,
                                                                        self.__last_lr)

        plt.title(title, pad=10)

        plt.plot(self.__costs)

        plt.ylabel('cost')
        plt.xlabel('epochs')

        accuracies = '\n'.join(['%s: %f' % (key, round(self.__accuracies[key], 7)) for key in ['Train', 'Test']])

        plt.text(0.65, 0.85, accuracies, ha='left', va='center', transform=plt.gca().transAxes, fontsize=13)

        fig = plt.gcf()
        plt.show()

        return fig

    @property
    def hyper_params(self):
        """
        Getter of the model's hyper parameters.
        :return: dictionary of the model's hyper parameters.
        """

        return dict(layers_dims=self.__layers_dims, activations=self.__activations, cost=self.__cost, lr=self.__last_lr,
                    accuracies=self.__accuracies)

    def describe(self):
        print(self)

    def __str__(self):
        """
        Returns a string representation of the model.
        :return: the string representing the model
        """

        layers_s = '\n'.join(['\t' + str(layer) for layer in self.__layers])
        cost_s = '\n\t' + str(self.__cost)
        params_s = '\ntotal %i Parameters\n' % sum([l.n_params for l in self.__layers if isinstance(l, WeightedLayer)])

        return PrintUtils.hyphen_line(35) + 'Model:\n' + layers_s + cost_s + params_s + PrintUtils.hyphen_line(25)
