import numpy as np
import operator


ACTUAL = 0
DATA = 1
PREDICTED = 2


def get_distance(a, b):
    return np.linalg.norm(a-b)


# Returns a dict with tuple as values:
# [ (actual, predicted) ]
def nearest_knn(train_data, test_data):
    distances = dict()
    for test_key in test_data:
        pass
