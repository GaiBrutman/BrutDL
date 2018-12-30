from abc import abstractmethod
import numpy as np


class Cost:
    def __init__(self, func_name):
        self.func_name = func_name.lower()

    @abstractmethod
    def _compute(self, AL, Y):
        pass

    def compute(self, AL, Y, get_node_cost=False):
        cost = self._compute(AL, Y)
        node_cost = np.mean(cost, axis=1)
        cost = np.mean(cost)

        assert cost.shape == ()

        return (cost, node_cost) if get_node_cost else cost

    @abstractmethod
    def derivative(self, AL, Y):
        pass

    def __str__(self):
        return '%s Cost' % self.func_name.upper()


class MeanSquaredError(Cost):
    def __init__(self):
        super(MeanSquaredError, self).__init__('mean squared error')

    def _compute(self, AL, Y, get_node_cost=False):
        return 0.5 * (Y - AL) ** 2

    def compute(self, AL, Y, get_node_cost=False):
        """
        Computes the MeanSquaredError cost of the model's output and the true labels.
        :param AL: output of the model
        :param Y: true labels
        :param get_node_cost: whether to return the cost for every output node as well
        :return: cost (mean of all nodes costs), (optional) cost for every output node
        """

        return super(MeanSquaredError, self).compute(AL, Y, get_node_cost)

    def derivative(self, AL, Y):
        """
        Computes the derivative of the MeanSquaredError cost function in respect to AL
        :param AL: output of the model
        :param Y: true labels
        :return: derivative of the cost in respect to AL
        """

        return -(Y - AL)


class BinaryCrossEntropy(Cost):
    def __init__(self, epsilon=1e-12):
        super(BinaryCrossEntropy, self).__init__('binary cross entropy')

        self.epsilon = epsilon

    def _compute(self, AL, Y, get_node_cost=False):
        AL = np.clip(AL, self.epsilon, 1 - self.epsilon)

        return - (Y * np.log(AL) + (1 - AL) * np.log(1 - AL))

    def compute(self, AL, Y, get_node_cost=False):
        """
        Computes the BinaryCrossEntropy cost of the model's output and the true labels.
        :param AL: output of the model
        :param Y: true labels
        :param get_node_cost: whether to return the cost for every output node as well
        :return: cost (mean of all nodes costs), (optional) cost for every output node
        """

        return super(BinaryCrossEntropy, self).compute(AL, Y, get_node_cost)

    def derivative(self, AL, Y):
        """
        Computes the derivative of the BinaryCrossEntropy cost function in respect to AL
        :param AL: output of the model
        :param Y: true labels
        :return: derivative of the cost in respect to AL
        """

        AL = np.clip(AL, self.epsilon, 1 - self.epsilon)
        divisor = np.maximum(AL * (1 - AL), self.epsilon)

        return (AL - Y) / divisor


class CategoricalCrossEntropy(Cost):
    def __init__(self, epsilon=1e-12):
        super(CategoricalCrossEntropy, self).__init__('categorical cross entropy')

        self.epsilon = epsilon

    def _compute(self, AL, Y, get_node_cost=False):
        AL = np.clip(AL, self.epsilon, 1 - self.epsilon)
        return -(Y * np.log(AL))

    def compute(self, AL, Y, get_node_cost=False):
        """
        Computes the CategoricalCrossEntropy cost of the model's output and the true labels.
        :param AL: output of the model
        :param Y: true labels
        :param get_node_cost: whether to return the cost for every output node as well
        :return: cost (mean of all nodes costs), (optional) cost for every output node
        """

        return super(CategoricalCrossEntropy, self).compute(AL, Y, get_node_cost)

    def derivative(self, AL, Y):
        """
        Computes the derivative of the CategoricalCrossEntropy cost function in respect to AL
        :param AL: output of the model
        :param Y: true labels
        :return: derivative of the cost in respect to AL
        """

        AL = np.clip(AL, self.epsilon, 1 - self.epsilon)
        return AL - Y


COSTS = {'mse': MeanSquaredError, 'binary_ce': BinaryCrossEntropy, 'categorical_ce': CategoricalCrossEntropy}
