import sys
import numpy as np
from adaboost import ada_boost
from nearest import nearest_knn
from neuralnets import neural_nets


ACTUAL = 0
DATA = 1
PREDICTED = 2


def calculate_accuracy(output):
    count = 0
    for compute in output:
        if compute[0] == compute[1]:
            count += 1
    print "Accuracy : ", (count * 1.0) / (len(output) * 1.0)


def get_normal(array):
    min_ = np.amin(array)
    max_ = np.amax(array)
    for i in range(len(array)):
        array[i] = (array[i] - min_)*1.0 / (max_ - min_)
    return array


# returns the following data structure.
# { photoID: { DATA:{0: [], 90: [], 180: [], 270: [] }, PREDICTED: None }
def populate_data(file_name):
    input_ = dict()
    train_ = open(file_name, "r")
    for line in train_.readlines():
        line = line.split()
        if line[0] not in input_:
            input_[line[0]] = dict()
            input_[line[0]][DATA] = dict()
            input_[line[0]][PREDICTED] = None
        degree = int(line[1])
        input_[line[0]][DATA][degree] = get_normal(np.array([int(j) for j in
                                                             line[2:]],
                                                            dtype=float))
    return input_


def main():
    mode = sys.argv[3]
    train_data = populate_data(sys.argv[1])
    test_data = populate_data(sys.argv[2])
    if mode == "nearest":
        calculate_accuracy(nearest_knn(train_data, test_data))
    elif mode == "adaboost":
        ada_boost(train_data, test_data, int(sys.argv[4]))
    elif mode == "nnet":
        calculate_accuracy(neural_nets(train_data, test_data, int(sys.argv[4])))


if __name__ == "__main__":
    main()

