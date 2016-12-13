import numpy as np
import random
import operator
from pprint import pprint
import math

# Returns a dict with tuple as values:
# { "photoID": (actual, predicted) }
def ada_boost(train_data, test_data, stump_count):
    # print("sbc")
    # for key in train_data:
    #     print(train_data[key][0])
    #     print(train_data[key][1][0])
    # degree = 0
    num_of_pairs = 10

    final_ensemble_dict = dict()


    for degree in [0,90,180,270]:
        # { 0: <stump_0>, 1: <stump_1>}
        # stump dict {'index1': <index_1>, 'index2': <index_2>, 'weight': <weight_of_stump>}
        ensemble_stumps_dict = dict()

        image_dict = init_image_dict(train_data)
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

        # print(ensemble_stumps_dict)

        # calculate alpha for weights
        # ensemble_stumps_dict = calcuate_alpha(ensemble_stumps_dict)
        # print(ensemble_stumps_dict)
        final_ensemble_dict[degree] = ensemble_stumps_dict
    # error_rate_class, error_rate_not_class = run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 90)
    # run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 90)
    # run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 180)
    # run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 270)
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

def test_and_classify(test_data, final_ensemble_dict):
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
                    else:
                        image_prediction_dict[file][degree][orient] -= weight
            prediction_list = sorted(image_prediction_dict[file][degree].items(), key=operator.itemgetter(1), reverse=True)
            if prediction_list[0][0] == degree:
                true_prediction_count += 1
            else:
                false_prediction_count += 1

    print("Accuracy")
    print(float(false_prediction_count)/(true_prediction_count+false_prediction_count))


# def run_test_on_decision_stumps(test_data_line, final_ensemble_dict):
#
#     degree_dict = dict()
#
#     for degree in final_ensemble_dict:
#         if degree not in degree_dict:
#             degree_dict[degree] = 0.0
#         for stump_num in final_ensemble_dict[degree]:
#             index1 = final_ensemble_dict[degree][stump_num]['index1']
#             index2 = final_ensemble_dict[degree][stump_num]['index2']
#
#             if test_data_line[index1] > test_data_line[index2]:
                






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

    # Might have to normalize again
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
    # total_incorrect = 0
    total = 0.0
    for file in train_data:
        # for orient in train_data[file][1]:
            # total += 1
            # Increment the weight for wrongly classified
        if train_data[file][1][orient_class][index1] <= train_data[file][1][orient_class][index2]:
            image_dict[file][orient_class] += image_dict[file][orient_class]*increment_factor
        else:
            image_dict[file][orient_class] += image_dict[file][orient_class] * decrement_factor
        total += image_dict[file][orient_class]
    # image_dict['TOTAL'] = total


    # Normalize
    for file in train_data:
        image_dict[file][orient_class] = image_dict[file][orient_class]/total

    return image_dict

def run_decision_stump(train_data, index1, index2, orient_class, image_dict):

    total = 0.0

    # image_dict = dict()
    image_dict['TOTAL'] = 0.0
    process_dict = dict()
    process_dict['TRUE'] = {"IN_CLASS": 0.0}
    process_dict['TRUE']['OUT_CLASS'] = 0.0
    process_dict['FALSE'] = {"IN_CLASS": 0.0}
    process_dict['FALSE']['OUT_CLASS'] = 0.0
    for file in train_data:
        if file not in image_dict:
            image_dict[file] = 1.0
        for degree in train_data[file][1]:
            total += image_dict[file]
            if train_data[file][1][degree][index1] > train_data[file][1][degree][index2]:
                if degree == orient_class:
                    process_dict['TRUE']['IN_CLASS'] += image_dict[file]
                else:
                    process_dict['TRUE']['OUT_CLASS'] += image_dict[file]
            else:
                if degree == orient_class:
                    process_dict['FALSE']['IN_CLASS'] += image_dict[file]
                else:
                    process_dict['FALSE']['OUT_CLASS'] += image_dict[file]
    image_dict['TOTAL'] = total
    print(process_dict['TRUE']['IN_CLASS'])
    print(process_dict['TRUE']['OUT_CLASS'])
    print(process_dict['FALSE']['IN_CLASS'])
    print(process_dict['FALSE']['OUT_CLASS'])
    print(total)
    error_rate_1 = float(process_dict['TRUE']['OUT_CLASS'])/image_dict['TOTAL']
    error_rate_2 = float(process_dict['TRUE']['IN_CLASS'])/image_dict['TOTAL']
    return error_rate_1, error_rate_2, image_dict
    print(error_rate_1)
    print(error_rate_2)
    print("orient_class:"+str(orient_class))
    # print(error_rate)
    # return error_rate

def run_decision_stump_degree(train_data, index1, index2, orient_class, image_dict):

    total = 0.0

    # image_dict = dict()
    # image_dict['TOTAL'] = 0.0
    process_dict = dict()
    process_dict['IN_CLASS'] = 0.0
    process_dict['OUT_CLASS'] = 0.0
    for file in train_data:
        # if file not in image_dict:
        #     image_dict[file] = 1.0
        # for degree in train_data[file][1]:
        total += image_dict[file][orient_class]
        if train_data[file][1][orient_class][index1] > train_data[file][1][orient_class][index2]:
            process_dict['IN_CLASS'] += image_dict[file][orient_class]
        else:
            process_dict['OUT_CLASS'] += image_dict[file][orient_class]
    # image_dict['TOTAL'] = total
    # print(process_dict['TRUE']['IN_CLASS'])
    # print(process_dict['TRUE']['OUT_CLASS'])
    # print(process_dict['FALSE']['IN_CLASS'])
    # print(process_dict['FALSE']['OUT_CLASS'])
    print(total)
    error_rate_1 = float(process_dict['OUT_CLASS'])/total
    # error_rate_2 = float(process_dict['IN_CLASS'])/image_dict['TOTAL']
    error_rate_2 = 1.0
    return error_rate_1, error_rate_2, image_dict
    print(error_rate_1)
    print(error_rate_2)
    print("orient_class:"+str(orient_class))
    # print(error_rate)
    # return error_rate
