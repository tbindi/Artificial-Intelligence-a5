import random
import math
import numpy as np


ACTUAL = 0
DATA = 1


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-1.0 * x))


def get_random():
    return random.uniform(0, 1)


def out_derivative(out):
    return out * (1 - out)


def init_random(N):
    return [get_random() for i in range(0, N)]


def get_result(list_):
    return np.array(list_).argmax() * 90


def get_delta(layer, errors):
    for i in range(0, len(layer)):
        neuron = layer[i]
        neuron["DELTA"] = 1.0 * errors[i] * out_derivative(neuron["OUTPUT"])
    return layer


class Neuron:

    def __init__(self):
        self.weights_layer = []
        self.weights_output = []
        self.constant = 0.01

    def activation(self, input_, index):
        return sum([i*j for i, j in zip(input_, self.weights_layer[index])])

    def get_output(self, input_, neuron_index):
        sum_ = 0.0
        for weight_index in range(len(self.weights_output[neuron_index])):
             sum_ += input_[weight_index]["OUTPUT"] * self.weights_output[
                 neuron_index][weight_index]
        return sum_

    def initialize_network(self, hidden_count):
        for i in range(0, hidden_count):
            self.weights_layer.append(init_random(192))
        for j in range(4):
            list_ = []
            for k in range(hidden_count):
                list_.extend(init_random(1))
            self.weights_output.append(list_)

    def feed_forward(self, hidden_count, input_row):
        hidden_layer = []
        out_ = []
        # Populate the hidden layer from the given input row
        for neuron_index in range(0, hidden_count):
            sum_ = self.activation(input_row, neuron_index)
            hidden_layer.append({"SUM": sum_, "OUTPUT": sigmoid(sum_)})
        # Populate the hidden layer from the given input row
        for neuron_index in range(4):
            out_.append(self.get_output(hidden_layer, neuron_index))
        output_layer = [{"SUM": i, "OUTPUT": sigmoid(i)} for i in out_]
        return output_layer, hidden_layer

    def backward_propagate(self, hidden_layer, output_layer, degree):
        errors = []
        # Output Layer Delta Calculation
        for neuron in output_layer:
            index = output_layer.index(neuron)
            if index * 90 == degree:
                errors.append(1 - neuron["OUTPUT"])
            else:
                errors.append(0 - neuron["OUTPUT"])
        output_layer = get_delta(output_layer, errors)
        errors = []
        # Hidden Layer Delta Calculation
        for j in range(len(hidden_layer)):
            error = 0.0
            for out_neuron in output_layer:
                ind = output_layer.index(out_neuron)
                error += (self.weights_output[ind][j] * out_neuron["DELTA"])
            errors.append(error)
        hidden_layer = get_delta(hidden_layer, errors)
        return output_layer, hidden_layer

    def update_weights(self, hidden_layer, output_layer, train_row):
        for neuron_index in range(len(hidden_layer)):
            for input_index in range(len(self.weights_layer[neuron_index])):
                self.weights_layer[neuron_index][input_index] += 0.0004 * \
                                                                 hidden_layer[neuron_index]["DELTA"] * train_row[input_index]
        for neuron_index in range(len(output_layer)):
            for input_index in range(len(self.weights_output[neuron_index])):
                self.weights_output[neuron_index][input_index] += 0.0004 * \
                                                                  output_layer[neuron_index]["DELTA"] * hidden_layer[input_index]["OUTPUT"]


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
