import random
from math import e as exp


ACTUAL = 0
DATA = 1


def sigmoid(x):
    return 1 / (1 + exp**(-x))


def get_random():
    return random.uniform(0, 1)


class Perceptron:

    def __init__(self):
        self.weights = [get_random() for i in range(0, 192)]
        self.constant = 0.01

    def feed_forward(self, input_):
        return sum([i*j for i, j in zip(input_, self.weights)])

    def train(self, input_):
        guess = self.feed_forward(input_)
        error = sigmoid(guess)


# Returns a dict with tuple as values:
# { testID:(actual, predicted) }
def neural_nets(train_data, test_data, hidden_count):
    p = Perceptron()
    for key in train_data:
        for degree in [0, 90, 180, 270]:
            input_ = train_data[key][DATA][degree]
