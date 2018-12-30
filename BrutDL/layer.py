from abc import abstractmethod, ABC


class Layer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
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
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def n_params(self):
        pass
