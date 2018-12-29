from abc import abstractmethod, ABC

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
