import sys
import numpy as np
from adaboost import ada_boost
from nearest import nearest_impl
from neuralnets import neural_nets


ACTUAL = 0
DATA = 1
PREDICTED = 2


def calculate_accuracy(output):
    pass


def display_accuracy(output):
    pass


# returns the following data structure.
# { photoID: { actual: class(0,90,180,270), data: np(), predicted: None }
def populate_data(file_name):
    input_ = dict()
    train_ = open(file_name, "r")
    for line in train_.readlines():
        line = line.split()
        input_[line[0]] = dict()
        input_[line[0]][ACTUAL] = int(line[1])
        input_[line[0]][DATA] = np.array([int(j) for j in line[2:]])
    return input_


def main():
    mode = sys.argv[3]
    train_data = populate_data(sys.argv[1])
    test_data = populate_data(sys.argv[2])
    if mode == "nearest":
        nearest_impl(train_data, test_data)
    elif mode == "adaboost":
        ada_boost(train_data, test_data, int(sys.argv[4]))
    elif mode == "nnet":
        neural_nets(train_data, test_data, int(sys.argv[4]))


if __name__ == "__main__":
    main()

