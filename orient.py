import sys
import numpy as np
from adaboost import ada_boost
from nearest import nearest_knn
from neuralnets import neural_nets


ACTUAL = 0
DATA = 1
PREDICTED = 2


def print_confusion_matrix(confusion_matrix):
    for i in confusion_matrix:
        print i


def calculate_accuracy(output, filename):
    count = 0
    confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for compute in output[0]:
        if compute[0] == compute[1]:
            count += 1
        confusion_matrix[compute[0]/90][compute[1]/90] += 1
    print "Accuracy : ", (count * 1.0) / (len(output[0]) * 1.0)
    print_confusion_matrix(confusion_matrix)
    file_ = open(filename, "w")
    for i in range(len(output[0])):
        file_.write(output[1][i]+" "+str(output[0][i][0]))
    file_.close()


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
        input_[line[0]][DATA][degree] = np.array([int(j) for j in line[2:]],
                                                 dtype=float)
    return input_


def main():
    mode = sys.argv[3]
    train_data = populate_data(sys.argv[1])
    test_data = populate_data(sys.argv[2])
    if mode == "nearest":
        calculate_accuracy(nearest_knn(train_data, test_data),
                           "nearest_output.txt")
    elif mode == "adaboost":
        ada_boost(train_data, test_data, int(sys.argv[4]))
    elif mode == "nnet":
        calculate_accuracy(neural_nets(train_data, test_data, int(sys.argv[
                                                                      4])),
                           "nnet_output.txt")


if __name__ == "__main__":
    main()

