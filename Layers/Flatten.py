from . import Base


class Flatten(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        num_samples = self.input_shape[0]
        flat = 1
        for i in range(1, len(self.input_shape)):
            flat *= self.input_shape[i]
        return input_tensor.reshape((num_samples, flat))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
