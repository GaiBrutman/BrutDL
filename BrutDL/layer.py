from abc import abstractmethod, ABC
from BrutDL.nn_utils_v1 import *


class Layer(ABC):
    @abstractmethod
    def fire(self, *args, **kwargs):
        pass

    @abstractmethod
    def back_prop(self, *args, **kwargs):
        pass


class WeightedLayer(Layer):
    def __init__(self):
        self.in_dim = None
        self.out_dim = None
        self.activation = None

    @abstractmethod
    def init_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def fire(self, *args, **kwargs):
        pass

    @abstractmethod
    def back_prop(self, *args, **kwargs):
        pass


class Activation(Layer):
    def __init__(self, func_name):
        super(Activation, self).__init__()
        self.func_name = func_name.lower()
        self.__fire_func = FORWARD_FUNCS[self.func_name]
        self.__back_prop_func = BACKWARD_FUNCS[self.func_name]
        self.cache = None

    def fire(self, Z):
        A = self.__fire_func(Z)
        self.cache = Z
        return A

    def back_prop(self, dA, lr):
        return self.__back_prop_func(dA, self.cache)

    def __str__(self):
        return '%s Activation Layer' % self.func_name.upper()
