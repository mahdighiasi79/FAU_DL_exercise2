import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next())
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, copy.deepcopy(self.label_tensor))
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(copy.deepcopy(self.label_tensor))
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        output_tensor = copy.deepcopy(input_tensor)
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor
