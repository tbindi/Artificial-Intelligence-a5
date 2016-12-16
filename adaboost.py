"""
This implementation of Adaboost uses Decision stumps as classifiers. We select the best of out of 10 decision stumps
based on which gives least error rate for each stump that needs to selected. This is done by selecting 2 random indexes(index1 and index2).
It is positive if value at index1 > value at index2 and negative otherwise. Each classifier is weighed based on the perfomance it
gave(based on error rate). Lesser the error rate more its weight. Finally, the adaboost outputs a list of classifiers for each class,
with a corresponding weight for each of them.

alpha calculation:

alpha = 1/2 ln((1 - err)/err)

data = data*e^alpha for misclassified
data = data*e^-alpha for correctly classified

final_ensemble_dict contains the list of classifiers for each degree/orientation with the index details.
test_and_classify() uses this final_ensemble_dict to check what is the majority weight to assign a class for each.

"""

import numpy as np
import random
import operator
from pprint import pprint
import math

# Returns a dict with tuple as values:
# { "photoID": (actual, predicted) }
def ada_boost(train_data, test_data, stump_count):
    num_of_pairs = 10
    final_ensemble_dict = dict()

    for degree in [0,90,180,270]:
        # { 0: <stump_0>, 1: <stump_1>}
        # stump dict {'index1': <index_1>, 'index2': <index_2>, 'weight': <weight_of_stump>}
        ensemble_stumps_dict = dict()

        image_dict = init_image_dict(train_data, True)
        for stump_num in range(0, stump_count):
            ensemble_stumps_dict[stump_num] = init_stumps_dict()

            decision_stump_dict_list = []
            # Try num_of_pairs of times to get the best stump
            for i in range(0, num_of_pairs):
                decision_stump_dict = dict()
                index1 = random.randint(0, 191)
                index2 = random.randint(0, 191)
                decision_stump_dict[0] = index1
                decision_stump_dict[1] = index2

                error_rate_class, error_rate_not_class, image_dict = run_decision_stump_degree(train_data, index1, index2, degree, image_dict)
                decision_stump_dict[2] = error_rate_class
                decision_stump_dict[3] = error_rate_not_class
                if error_rate_class < error_rate_not_class:
                    decision_stump_dict[4] = error_rate_class
                else:
                    decision_stump_dict[4] = error_rate_not_class
                decision_stump_dict_list.append(decision_stump_dict)
            print(decision_stump_dict_list)

            # Sort to get the best Stump
            decision_stump_dict_list = sorted(decision_stump_dict_list, key=operator.itemgetter(4))

            cur_decision_stump_dict = decision_stump_dict_list[0]
            cur_decision_stump_dict[4] = calcuate_alpha_decision_stump_dict(cur_decision_stump_dict[4])
            # decision_stump_dict_list[0] has the best stump currently
            # Normalize image_dict
            image_dict = normalize_image_dict_weights(train_data, cur_decision_stump_dict, image_dict, degree)

            # Set the stump at this stump_num
            ensemble_stumps_dict[stump_num]['index1'] = cur_decision_stump_dict[0]
            ensemble_stumps_dict[stump_num]['index2'] = cur_decision_stump_dict[1]
            ensemble_stumps_dict[stump_num]['weight'] = cur_decision_stump_dict[4]
            if cur_decision_stump_dict[4] == cur_decision_stump_dict[3]: # By default is in_class true
                ensemble_stumps_dict[stump_num]['in_class'] = False

        final_ensemble_dict[degree] = ensemble_stumps_dict
    print("FINAL")
    pprint(final_ensemble_dict)

    test_and_classify(test_data, final_ensemble_dict)

def init_image_dict(train_data, is_degree=None):
    image_dict = dict()
    total = 0
    if not is_degree:
        total = len(train_data)*4
    else:
        total = len(train_data)
    for file in train_data:
        if not file in image_dict:
            image_dict[file] = dict()
        for orient in train_data[file][1]:
            image_dict[file][orient] = 1.0/total
    return image_dict

def init_confusion_row():
    row = dict()
    for degree in [0, 90, 180, 270]:
        row[degree] = 0
    return row

def test_and_classify(test_data, final_ensemble_dict):

    adaboost_file = open("adaboost_output.txt", 'w')
    confusion_matrix = dict()
    for degree in [0,90,180,270]:
        confusion_matrix[degree] = init_confusion_row()

    image_prediction_dict = dict()
    true_prediction_count = 0
    false_prediction_count = 0
    for file in test_data:
        if file not in image_prediction_dict:
            image_prediction_dict[file] = {}
        for degree in test_data[file][1]:
            image_prediction_dict[file][degree] = {0:0.0, 90:0.0, 180:0.0, 270:0.0}
            for orient in final_ensemble_dict:
                for classifier_num in final_ensemble_dict[orient]:
                    index1 = final_ensemble_dict[orient][classifier_num]['index1']
                    index2 = final_ensemble_dict[orient][classifier_num]['index2']
                    weight = final_ensemble_dict[orient][classifier_num]['weight']
                    if test_data[file][1][degree][index1] > test_data[file][1][degree][index2]:
                        image_prediction_dict[file][degree][orient] += weight
                    # else:
                    #     image_prediction_dict[file][degree][orient] -= weight
            prediction_list = sorted(image_prediction_dict[file][degree].items(), key=operator.itemgetter(1), reverse=True)
            adaboost_file.write(file+" "+str(prediction_list[0][0])+"\n")
            if prediction_list[0][0] == degree:
                true_prediction_count += 1
            else:
                false_prediction_count += 1
            confusion_matrix[degree][prediction_list[0][0]] += 1

    adaboost_file.close()
    print("Accuracy:")
    print(float(true_prediction_count)/(true_prediction_count+false_prediction_count))
    print("Confusion Matrix:")
    print_confusion_matrix(confusion_matrix)

def print_confusion_matrix(confusion_matrix):
    matrix_str = "     "

    for row in confusion_matrix:
        matrix_str += str(row) + "   "
    matrix_str += "\n"

    for row in confusion_matrix:
        matrix_str += str(row) + "   "
        for col in confusion_matrix[row]:
            matrix_str = matrix_str + str(confusion_matrix[row][col]) + "   "
        matrix_str = matrix_str + "\n"
    print(matrix_str)


def calcuate_alpha_decision_stump_dict(error_rate):
    # error_rate = decision_stump_dict[4]
    alpha = 0.5*math.log((1-error_rate)/error_rate)
    return alpha


def calcuate_alpha(ensemble_stumps_dict):
    total = 0.0
    for stump_num in ensemble_stumps_dict:
        total += ensemble_stumps_dict[stump_num]['weight']

    for stump_num in ensemble_stumps_dict:
        ensemble_stumps_dict[stump_num]['weight'] = 1 - (ensemble_stumps_dict[stump_num]['weight']/total)

    return ensemble_stumps_dict


def init_stumps_dict():
    # stump dict {'index1': <index_1>, 'index2': <index_2>, 'weight': <weight_of_stump>}
    stumps_dict = dict()
    stumps_dict['index1'] = 0
    stumps_dict['index2'] = 0
    stumps_dict['weight'] = 0.0
    stumps_dict['in_class'] = True
    return stumps_dict

def normalize_image_dict_weights(train_data, decision_stump_dict, image_dict, orient_class):
    increment_factor = math.pow(math.e, decision_stump_dict[4])
    decrement_factor = math.pow(math.e, -decision_stump_dict[4])
    index1 = decision_stump_dict[0]
    index2 = decision_stump_dict[1]
    total = 0.0
    for file in train_data:
         # Increment the weight for wrongly classified
        if train_data[file][1][orient_class][index1] <= train_data[file][1][orient_class][index2]:
            image_dict[file][orient_class] += image_dict[file][orient_class]*increment_factor
        else:
            image_dict[file][orient_class] += image_dict[file][orient_class] * decrement_factor
        total += image_dict[file][orient_class]

    # Normalize
    for file in train_data:
        image_dict[file][orient_class] = image_dict[file][orient_class]/total

    return image_dict

def run_decision_stump_degree(train_data, index1, index2, orient_class, image_dict):

    total = 0.0
    process_dict = dict()
    process_dict['IN_CLASS'] = 0.0
    process_dict['OUT_CLASS'] = 0.0
    for file in train_data:
        total += image_dict[file][orient_class]
        if train_data[file][1][orient_class][index1] > train_data[file][1][orient_class][index2]:
            process_dict['IN_CLASS'] += image_dict[file][orient_class]
        else:
            process_dict['OUT_CLASS'] += image_dict[file][orient_class]
    # print(total)
    error_rate_1 = float(process_dict['OUT_CLASS'])/total
    error_rate_2 = 1.0
    return error_rate_1, error_rate_2, image_dict
