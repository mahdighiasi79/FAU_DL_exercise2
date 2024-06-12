import numpy as np


class Constant:

    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.zeros(fan_in * fan_out).reshape(weights_shape) + self.value


class UniformRandom:

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, fan_in * fan_out).reshape(weights_shape)


class Xavier:

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        standard_deviation = np.power(2 / (fan_in + fan_out), 0.5)
        return np.random.normal(0, standard_deviation, fan_in * fan_out).reshape(weights_shape)


class He:

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        standard_deviation = np.power(2 / fan_in, 0.5)
        return np.random.normal(0, standard_deviation, fan_in * fan_out).reshape(weights_shape)
