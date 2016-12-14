import random
from math import e as exp


ACTUAL = 0
DATA = 1


def sigmoid(x):
    return 1 / (1 + exp**(-x))


def get_random():
    return random.uniform(0, 1)


def init_random(N):
    return [get_random() for i in range(0, N)]


class Neuron:

    def __init__(self):
        self.weights_layer = []
        self.weights_output = []
        self.output = []
        self.constant = 0.01

    def activation(self, input_, index):
        return sum([i*j for i, j in zip(input_, self.weights_layer[index])])

    def get_output(self, input_, index):
        return [input_*j for j in self.weights_output[index]]

    def initialize_network(self, hidden_count):
        for i in range(0, hidden_count):
            self.weights_layer.append(init_random(192))
            self.weights_output.append(init_random(4))
        for i in range(0, 4):
            self.output.append(0.0)

    def feed_forward(self, hidden_count, input_row, output_layer, degree):
        hidden_layer = []
        out_ = []
        # Populate the hidden layer from the given input row
        for index in range(0, hidden_count):
            sum_ = self.activation(input_row, index)
            hidden_layer.append([sum_, sigmoid(sum_)])
        # Populate the output layer from the given hidden layer
        for index in range(hidden_count):
            out_.append(self.get_output(hidden_layer[index][1], index))
        output_layer = [[sum(i), sigmoid(sum(i))] for i in zip(*out_)]
        for index in range(4):
            error = output_layer[index][1]
            if index * 90 == degree:
                output_layer[index][1] = 1 - error
            else:
                output_layer[index][1] = 0 - error
        return output_layer, hidden_layer

    def backward_propagate(self, hidden_layer, output_layer):
        for error in output_layer:
            pass


# Returns a dict with tuple as values:
# { testID:(actual, predicted) }
def neural_nets(train_data, test_data, hidden_count):
    p = Neuron()
    p.initialize_network(hidden_count)
    output_layer = []
    for row in train_data:
        for degree in [0, 90, 180, 270]:
            input_row = train_data[row][DATA][degree]
            output_layer, hidden_layer = p.feed_forward(hidden_count,
                                                        input_row,
                                                        output_layer, degree)
            # p.backward_propagate(hidden_layer, output_layer)
