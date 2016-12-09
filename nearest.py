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
    # { test_key: [(distance, degree, train_key)]
    distances = dict()
    for test_key in test_data:
        test_value = test_data[test_key][DATA].values()[0]
        temp_dist = []
        for train_key in train_data:
            for degree_key in train_data[train_key][DATA]:
                value = train_data[train_key][DATA][degree_key]
                dist = get_distance(value, test_value)
                if len(temp_dist) <= 10:
                    temp_dist.append((dist, degree_key, train_key))
                    sorted(temp_dist, key=operator.itemgetter(0))
                else:
                    if dist < temp_dist[-1][0]:
                        temp_dist.append((dist, degree_key, train_key))
                        sorted(temp_dist, key=operator.itemgetter(0))
                        temp_dist.pop()
        distances[test_key] = temp_dist

    vote = dict()
    result = []
    for test_key in distances:
        each_vote = dict()
        for value in distances[test_key]:
            if value[1] in each_vote:
                each_vote[value[1]] += 1
            else:
                each_vote[value[1]] = 1
        vote[test_key] = max(each_vote.iteritems(), key=operator.itemgetter(
            1))[0]
        result.append((vote[test_key], test_data[test_key][DATA].keys()[0]))
    return result
