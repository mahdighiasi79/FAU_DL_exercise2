import math
import numpy as np

from . import Base


class Pooling(Base.BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        batch_size, num_channels, x_in, y_in = input_tensor.shape
        x_out = math.floor((x_in - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        y_out = math.floor((y_in - self.pooling_shape[1]) / self.stride_shape[1]) + 1
        output_tensor = np.zeros((batch_size, num_channels, x_out, y_out))

        for i in range(batch_size):
            for j in range(num_channels):
                for k in range(x_out):
                    x_start_index = k * self.stride_shape[0]
                    x_end_index = x_start_index + self.pooling_shape[0]

                    for l in range(y_out):
                        y_start_index = l * self.stride_shape[1]
                        y_end_index = y_start_index + self.pooling_shape[1]

                        output_tensor[i][j][k][l] = np.max(input_tensor[i, j, x_start_index:x_end_index, y_start_index:y_end_index])
        return output_tensor

    def backward(self, error_tensor):
        pass
