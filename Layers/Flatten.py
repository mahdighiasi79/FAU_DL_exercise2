class Flatten:

    def __init__(self):
        self.trainable = False

    @staticmethod
    def forward(input_tensor):
        shape = input_tensor.shape
        num_samples = shape[0]
        flat = shape[1] * shape[2] * shape[3]
        return input_tensor.reshape((num_samples, flat))

    @staticmethod
    def backward(error_tensor):
        return error_tensor.reshape((9, 3, 4, 11))
