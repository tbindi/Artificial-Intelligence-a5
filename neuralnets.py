import random
from math import e as exp
import numpy as np


ACTUAL = 0
DATA = 1


def sigmoid(x):
    return 1.0 / (1.0 + exp**(-x))


def get_random():
    return random.uniform(0, 1)


def out_derivative(out):
    return out * (1 - out)


def init_random(N):
    return [get_random() for i in range(0, N)]


def get_result(list_):
    return np.array(list_).argmax() * 90


class Neuron:

    def __init__(self):
        self.weights_layer = []
        self.weights_output = []
        self.constant = 0.01

    def activation(self, input_, index):
        return sum([i*j for i, j in zip(input_, self.weights_layer[index])])

    def get_output(self, input_, index):
        return [input_*j for j in self.weights_output[index]]

    def initialize_network(self, hidden_count):
        for i in range(0, hidden_count):
            self.weights_layer.append(init_random(192))
            self.weights_output.append(init_random(4))

    def feed_forward(self, hidden_count, input_row):
        hidden_layer = []
        out_ = []
        # Populate the hidden layer from the given input row
        for index in range(0, hidden_count):
            sum_ = self.activation(input_row, index)
            hidden_layer.append({"SUM": sum_, "OUTPUT": sigmoid(sum_)})
        # Populate the output layer from the given hidden layer
        for index in range(hidden_count):
            out_.append(self.get_output(hidden_layer[index]["OUTPUT"], index))
        output_layer = [{"SUM": sum(i), "OUTPUT": sigmoid(sum(i))} for i in \
                zip(*out_)]
        return output_layer, hidden_layer

    def get_delta(self, layer, errors):
        for i in range(0, len(layer)):
            neuron = layer[i]
            neuron["DELTA"] = errors[i] * out_derivative(neuron["OUTPUT"])
        return layer

    def backward_propagate(self, hidden_layer, output_layer, degree):
        errors = []
        # Output Layer Delta Calculation
        for neuron in output_layer:
            index = output_layer.index(neuron)
            if index * 90 == degree:
                errors.append(1 - neuron["OUTPUT"])
            else:
                errors.append(0 - neuron["OUTPUT"])
        output_layer = self.get_delta(output_layer, errors)
        errors = []
        # Hidden Layer Delta Calculation
        for i in range(0, len(hidden_layer)):
            error = 0.0
            for out_neuron in output_layer:
                j = output_layer.index(out_neuron)
                error += (self.weights_output[i][j] * out_neuron["DELTA"])
            errors.append(error)
        hidden_layer = self.get_delta(hidden_layer, errors)
        return output_layer, hidden_layer

    def update_weights(self, hidden_layer, output_layer, train_row):
        for i in range(0, len(self.weights_layer)):
            for j in range(0, len(self.weights_layer[i])):
                self.weights_layer[i][j] += 0.01 * hidden_layer[i]["DELTA"] * \
                                            train_row[j]
        for i in range(0, len(self.weights_output)):
            for j in range(0, len(self.weights_output[i])):
                self.weights_output[i][j] += 0.01 * output_layer[i]["DELTA"] * \
                hidden_layer[i]["OUTPUT"]


# Returns a dict with tuple as values:
# [ (actual, predicted) ]
def neural_nets(train_data, test_data, hidden_count):
    p = Neuron()
    p.initialize_network(hidden_count)
    for row in train_data:
        for degree in [0, 90, 180, 270]:
            input_row = train_data[row][DATA][degree]
            output_layer, hidden_layer = p.feed_forward(hidden_count, input_row)
            output_layer, hidden_layer = p.backward_propagate(hidden_layer,
                                                     output_layer, degree)
            p.update_weights(hidden_layer, output_layer, input_row)
    result = []
    for row in test_data:
        degree = test_data[row][DATA].keys()[0]
        test_row = test_data[row][DATA].values()[0]
        out_, hidden_ = p.feed_forward(hidden_count, test_row)
        print "PREDICTED:", get_result(out_), " ACTUAL:", degree, " ", out_
        result.append((degree, get_result(out_)))
    return result
