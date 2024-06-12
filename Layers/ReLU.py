import numpy as np
# import Base


class ReLU:

    def __init__(self):
        # super().__init__()
        self.trainable = False
        self.input_tensor = np.array([])
        self.type = "ReLU"

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = (input_tensor > 0).astype(float)
        output_tensor *= input_tensor
        return output_tensor

    def backward(self, error_tensor):
        relu_derivative = (self.input_tensor > 0).astype(float)
        error_tensor *= relu_derivative
        return error_tensor
