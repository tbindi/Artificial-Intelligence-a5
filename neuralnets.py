"""

Steps involved:

1. Train the network:
    a. Initialize network: Build a neural network with the hidden neuron count.
    b. Forward Propagation: This calculates the output of each neuron in each layer as a sum of weights * inputs for that neuron.
    c. Back Propagation: This is a way in which the error at each layer is calculated for each neuron.
    d. Update Weights: This is calculated considering the learning rate, delta value and the previous received at each layer and neuron.
    d. Activation: Calculation of a output which quantifies the value at that position.

2. Predict the data:
    Taking the test data and running it using the forward propagation will give a output value at the end.
    The output neuron for a class which throws the maximum probability is the predicted class for the data row.


Explanation of different functions:

Sigmoid function: 1.0 / (1.0 + math.exp(-1.0 * x))
Activation funtion: sum of weight * input
Transfer Derivative:  output * (1.0 - output)
Error function: for output layer: (expected_output - actual_output) * transfer_derivative,
                for hidden layer: (weight*error)*transfer_derivative.
learning rate: It is a parameter which controls how much faster the weights can converge onto the final set.


"""


import random
import math
import numpy as np


ACTUAL = 0
DATA = 1


def get_normal(array):
    for i in range(len(array)):
        array[i] = (array[i]) / sum(array)
    return array


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-1.0 * x))


def get_random():
    return random.uniform(0.001, 0.009)


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
        return sum([sigmoid(i)*j for i, j in zip(input_, self.weights_layer[
            index])])

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
                neuron["OUTPUT"] = 1 - neuron["OUTPUT"]
            else:
                errors.append(0 - neuron["OUTPUT"])
                neuron["OUTPUT"] = 0 - neuron["OUTPUT"]
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
        learning_rate = 0.05
        for neuron_index in range(len(hidden_layer)):
            for input_index in range(len(self.weights_layer[neuron_index])):
                self.weights_layer[neuron_index][input_index] += learning_rate * \
                                                                 hidden_layer[neuron_index]["DELTA"] * train_row[input_index]
        for neuron_index in range(len(output_layer)):
            for input_index in range(len(self.weights_output[neuron_index])):
                self.weights_output[neuron_index][input_index] += learning_rate * \
                                                                  output_layer[neuron_index]["DELTA"] * hidden_layer[input_index]["OUTPUT"]


# Returns a dict with tuple as values:
# [ (actual, predicted) ]
def neural_nets(train_data, test_data, hidden_count):
    p = Neuron()
    p.initialize_network(hidden_count)
    for row in train_data:
        for degree in [0, 90, 180, 270]:
            input_row = get_normal(train_data[row][DATA][degree])
            output_layer, hidden_layer = p.feed_forward(hidden_count, input_row)
            output_layer, hidden_layer = p.backward_propagate(hidden_layer,
                                                         output_layer, degree)
            p.update_weights(hidden_layer, output_layer, input_row)
    result = []
    keys = []
    for row in test_data:
        degree = test_data[row][DATA].keys()[0]
        test_row = get_normal(test_data[row][DATA].values()[0])
        out_, hidden_ = p.feed_forward(hidden_count, test_row)
        result.append((degree, get_result(out_)))
        keys.append(row)
    return result, keys
