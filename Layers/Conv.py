import copy
import math
import numpy as np
from . import Initializers as Init
from . import Base


class Conv(Base.BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        weights_shape = list(copy.deepcopy(convolution_shape)).insert(0, num_kernels)
        if len(convolution_shape) == 2:
            convolution_size = convolution_shape[0] * convolution_shape[1]
        else:
            convolution_size = convolution_shape[0] * convolution_shape[1] * convolution_shape[2]

        self.weights = Init.UniformRandom.initialize(weights_shape, num_kernels, convolution_size)
        self.bias = Init.UniformRandom.initialize((num_kernels,), num_kernels, 1)
        self._gradient_weights = None
        self._gradient_biases = None

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, value):
        self._gradient_weights = value

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def get_gradient_biases(self):
        return self._gradient_biases

    def set_gradient_biases(self, value):
        self._gradient_biases = value

    gradient_biases = property(get_gradient_biases, set_gradient_biases)

    @staticmethod
    def padding(n, s, f):
        a = ((s - 1) * n) + f
        if (a - s) % 2 == 0:
            right_padding = left_padding = (a - s) / 2
        elif s > 1:
            right_padding = left_padding = (a - s + 1) / 2
        else:
            right_padding = a / 2
            left_padding = (a / 2) - 1
        return int(right_padding), int(left_padding)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]
        x_size = input_tensor.shape[2]
        x_filter_size = self.convolution_shape[1]
        if len(self.stride_shape) == 1:
            x_stride = y_stride = self.stride_shape[0]
        else:
            x_stride, y_stride = self.stride_shape

        if len(input_tensor.shape) == 3:
            x_right_padding, x_left_padding = self.padding(x_size, 1, x_filter_size)
            right_padding = np.zeros((batch_size, num_channels, x_right_padding))
            left_padding = np.zeros((batch_size, num_channels, x_left_padding))
            convolution_input = np.concatenate((left_padding, input_tensor, right_padding), axis=2)
            n_x = math.floor((x_size + x_right_padding + x_left_padding - x_filter_size) / x_stride) + 1

            # performing convolution
            output = []
            for i in range(self.num_kernels):
                channel = []
                for j in range(n_x):
                    start_index = j * x_stride
                    end_index = start_index + x_filter_size
                    convolve = convolution_input[:, :, start_index:end_index] * self.weights[i]
                    convolve = np.sum(convolve, axis=2, keepdims=False)
                    convolve = np.sum(convolve, axis=1, keepdims=False)
                    channel.append(convolve)
                channel = np.array(channel).reshape((batch_size, n_x))
                channel += self.bias[i]
                output.append(channel)
            output = np.array(output).reshape((batch_size, self.num_kernels, n_x))

        else:
            y_size = input_tensor.shape[3]
            y_filter_size = self.convolution_shape[2]
            x_right_padding, x_left_padding = self.padding(x_size, 1, x_filter_size)
            y_right_padding, y_left_padding = self.padding(y_size, 1, y_filter_size)
            right_padding_x = np.zeros((batch_size, num_channels, x_right_padding, y_size))
            left_padding_x = np.zeros((batch_size, num_channels, x_left_padding, y_size))
            convolution_input = np.concatenate((left_padding_x, input_tensor, right_padding_x), axis=2)
            right_padding_y = np.zeros((batch_size, num_channels, x_size + x_right_padding + x_left_padding, y_right_padding))
            left_padding_y = np.zeros((batch_size, num_channels, x_size + x_right_padding + x_left_padding, y_left_padding))
            convolution_input = np.concatenate((left_padding_y, convolution_input, right_padding_y), axis=3)
            n_x = math.floor((x_size + x_right_padding + x_left_padding - x_filter_size) / x_stride) + 1
            n_y = math.floor((y_size + y_right_padding + y_left_padding - y_filter_size) / y_stride) + 1

            # performing convolution
            output = []
            for i in range(self.num_kernels):
                channel = []
                for j in range(n_x):
                    x_out = []
                    x_start_index = j * x_stride
                    x_end_index = x_start_index + x_filter_size
                    for k in range(n_y):
                        y_start_index = k * y_stride
                        y_end_index = y_start_index + y_filter_size
                        convolve = convolution_input[:, :, x_start_index:x_end_index, y_start_index:y_end_index] * self.weights[i]
                        convolve = np.sum(convolve, axis=3, keepdims=False)
                        convolve = np.sum(convolve, axis=2, keepdims=False)
                        convolve = np.sum(convolve, axis=1, keepdims=False)
                        x_out.append(convolve)
                    x_out = np.array(x_out).reshape((batch_size, n_y))
                    channel.append(x_out)
                channel = np.array(channel).reshape((batch_size, n_x, n_y))
                channel += self.bias[i]
                output.append(channel)
            output = np.array(output).reshape((batch_size, self.num_kernels, n_x, n_y))

        return output

    def backward(self, error_tensor):
        pass

    def initialize(self, weights_initializer, bias_initializer):
        pass
