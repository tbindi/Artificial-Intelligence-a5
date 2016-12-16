# KNN working flow :-
# Loop through all the test points and calculate the distance between each test point with all the
# train points. Create a dictionary which keeps track of all the distances calculated. Variate k to
# find the best accuracy
# We used the np.linalg.norm() function to calculate the distance which also normalizes the data.
# We tried various distance functions  Euclidean, Manhattan,etc but
# Euclidean distance gave the
# best result so we tried to optimize this further.
#
# Limitations : -
# Computation time. It is taking around 20-30 mins just to calculate all the possible distances
# and then compute the predicted class of the test point. We checked the different between the
# accuracies given by normalized as well as unnormalized data and it was around the same.
# Normalized data is a little faster for computation.
# Getting an accuracy of 64 when k is 77
#   K               Accuracy
#   3                   59
#   5                   61
#   31                  61.18
#   77                  64.79
#   91                  64.79
#
#

# Computation takes around 20 mins.



import numpy as np
import operator
from operator import itemgetter
from collections import Counter
import cPickle



ACTUAL = 0
DATA = 1
PREDICTED = 2

#number of k
k = 5

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
                #Find k neighbours
                if len(temp_dist) <= k:
                    temp_dist.append((dist, degree_key, train_key))
                    sorted(temp_dist, key=operator.itemgetter(0))
                else:
                    if dist < temp_dist[-1][0]:
                        temp_dist.append((dist, degree_key, train_key))
                        sorted(temp_dist, key=operator.itemgetter(0))
                        temp_dist.pop()
        distances[test_key] = temp_dist

    # with open('distances_k-250.pickle', 'w') as f:  # Python 3: open(..., 'wb')
    #     cPickle.dump(distances, f)



    # with open('distances_k-250.pickle') as g:
    #     distances = cPickle.load(g)

    result = []
    for test_key in distances:
        temp = distances.get(test_key)
        temp.sort(key=itemgetter(0))
        temp = temp[:k]
        predicted_key = Counter(map(itemgetter(1), temp)).most_common(1)[0][0]
        result.append((test_data[test_key][DATA].keys()[0], predicted_key))

    return result
