from __future__ import print_function

import math

import matplotlib.pyplot as plt

from BrutDL.data import Data
from BrutDL.layer import *
from BrutDL.nn_utils_v1 import COST_FUNCS, LOSS_DERIV_FUNCS


class PrintModelProgress:
    @staticmethod
    def line_print(*args):
        print('\r', end='', flush=True)
        print(''.join(args), end='', flush=True)

    @staticmethod
    def progress_str(i, iter_max=1, print_max=30):
        progress_num = int(print_max * (i + 1) / iter_max)
        progress_str = '[' + '=' * progress_num + '>' + ' ' * (print_max - progress_num) + ']'
        return progress_str

    @staticmethod
    def print_progress(i, iter_max, epoch, cost, *args):
        """
        Prints the progress of the training in the same line.
        :param i: current iteration
        :param iter_max: iteration limit
        :param epoch: current epoch
        :param cost: cost value
        :param args: additional arguments (Strings).
        :return: None
        """

        progress_visual = PrintModelProgress.progress_str(i, iter_max=iter_max)
        s = 'epoch %i ' % epoch + progress_visual + ' Cost: %f' % float(cost)

        PrintModelProgress.line_print(s, *[', ' + a for a in args if a])


class Model:
    def __init__(self, input_shape, layers=None, loss='mse'):
        """
        Initialises of the class

        :param input_shape: the shape of the input data
        :param layers: list of model layers
        :param loss: the loss function to use in back propagation
        """

        super(Model, self).__init__()

        self.__layers_dims = [input_shape]  # List of array dimensions (int) in the model

        self.__layers = []

        self.__activations = []  # List of activation (str) functions applied in the model

        self.__loss = loss  # Loss function (str)

        self.__costs = []  # Keep track of cost

        self.__last_epochs = 0  # Last number of train epochs
        self.__last_train_size = 0  # Last number of training samples
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
            A = l.fire(A)

        return A

    def backward(self, AL, Y, lr):
        """
        Runs backward propagation, updates parameters for all layers

        :param AL: last post-activation value (from forward propagation)
        :param Y: data correct labels
        :param lr: the backward propagation learning rate
        :return: None
        """

        dAL = LOSS_DERIV_FUNCS[self.__loss](AL, Y)

        dA = dAL
        for l in reversed(self.__layers):
            dA = l.back_prop(dA, lr)

    def train(self, data=None, train_data=(None, None), val_data=(None, None), lr=1e-3, epochs=7, batch_size=64,
              verbose=True, lr_descent=False):
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
        :return: None
        """

        assert data is not None or Data.is_valid(train_data)

        if data is None:
            data = Data(train_data=train_data, val_data=val_data, start_dim=self.__layers_dims[0],
                        end_dim=self.__layers_dims[-1])

        assert isinstance(data, Data)

        self.__last_train_size = data.train_size
        self.__last_lr = lr
        self.__last_epochs += epochs

        # Loop over epochs
        for epoch in range(epochs):
            lr = self.__last_lr

            batch_cost = np.zeros(data.end_dim)
            # Loop over epochs
            for j, (min_batch_x, min_batch_y) in enumerate(data.mini_batches(batch_size)):
                # Forward propagation
                AL = self.forward(min_batch_x)

                # Compute cost
                cost, nodes_cost = COST_FUNCS[self.__loss](AL, min_batch_y)
                batch_cost += nodes_cost / data.n_batches

                # Back propagation
                self.backward(AL, min_batch_y, lr)

                # Print the training progress
                if verbose:
                    PrintModelProgress.print_progress(j + 1, data.n_batches, epoch, cost,
                                                      'Lr: %f' % lr if lr_descent else '')

                if lr_descent:
                    lr = (self.__last_lr / 2.) * math.cos((j / data.n_batches) * math.pi) + (self.__last_lr / 2.)

            self.__costs.append(batch_cost)

            # Print progress of each epoch
            if verbose:
                # If validation data exists, print validation cost
                if Data.is_valid(data.val_set):

                    val_cost, _ = COST_FUNCS[self.__loss](self.forward(data.X_val), data.Y_val)
                else:
                    val_cost = None

                PrintModelProgress.print_progress(j + 1, data.n_batches, epoch, np.mean(batch_cost),
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

    def classifier_accuracy(self, dataset, type_='Test'):
        """
        Calculates the accuracy of the model on given data.

        :param dataset: tuple of Data to be passed and labels for calculating the accuracy
        :param type_: type of data given. Used for saving the accuracy value in the right spot
        :return: the accuracy value of the model on the given data.
        """

        assert type_ in self.__accuracies

        (X, Y) = dataset

        # Forward propagation
        m = X.shape[-1]

        p = np.argmax(self.predict(X), axis=0)
        r = np.argmax(Y, axis=0)

        print('pred:', p[:10])
        print('true:', r[:10])

        acc = float(np.sum((p == r) / m))
        self.__accuracies[type_] = acc
        print('{} Accuracy: {}\n'.format(type_, acc))

        return acc

    def plot_cost(self):
        """
        Plots the cost history of the model, also shows different model parameters and accuracies.
        :return: the plot figure
        """

        title = 'layers = {}\n' \
                '         {}\n' \
                'size = {}, EPOCHS = {}, Learning rate = {}'.format(self.__layers_dims, self.__activations,
                                                                    self.__last_train_size, self.__last_epochs,
                                                                    self.__last_lr)

        plt.title(title)

        m = max(self.__last_epochs / len(self.__costs), 1) if self.__last_epochs and self.__costs else 0

        plt.plot(self.__costs)

        plt.ylabel('cost')
        plt.xlabel('iterations (per {})'.format(round(m, 1)))

        accuracies = 'Train: {}\nTest: {}'.format(str(self.__accuracies['Train'])[:14],
                                                  str(self.__accuracies['Test'])[:14])

        plt.text(0.8, 0.9, accuracies, ha='center', va='center', transform=plt.gca().transAxes)

        fig = plt.gcf()
        plt.show()

        return fig

    @property
    def hyper_params(self):
        """
        Getter of the model's hyper parameters.
        :return: dictionary of the model's hyper parameters.
        """

        return dict(layers_dims=self.__layers_dims, activations=self.__activations, loss=self.__loss, lr=self.__last_lr,
                    epoches=self.__last_epochs, accuracies=self.__accuracies)

    def __str__(self):
        """
        Returns a string representation of the model.
        :return: the string representing the model
        """

        return 'Model:\n' + '\n'.join(['\t' + str(layer) for layer in self.__layers])