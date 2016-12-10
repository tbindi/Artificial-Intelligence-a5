import numpy as np
import random
import operator
from pprint import pprint

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
        image_dict = dict()
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

            # decision_stump_dict_list[0] has the best stump currently
            # Normalize image_dict
            image_dict = normalize_image_dict_weights(train_data, decision_stump_dict_list[0], image_dict)

            # Set the stump at this stump_num
            ensemble_stumps_dict[stump_num]['index1'] = decision_stump_dict_list[0][0]
            ensemble_stumps_dict[stump_num]['index2'] = decision_stump_dict_list[0][1]
            ensemble_stumps_dict[stump_num]['weight'] = decision_stump_dict_list[0][4]
            if decision_stump_dict_list[0][4] == decision_stump_dict_list[0][3]: # By default is in_class true
                ensemble_stumps_dict[stump_num]['in_class'] = False

        # print(ensemble_stumps_dict)

        # calculate alpha for weights
        ensemble_stumps_dict = calcuate_alpha(ensemble_stumps_dict)
        # print(ensemble_stumps_dict)
        final_ensemble_dict[degree] = ensemble_stumps_dict
    # error_rate_class, error_rate_not_class = run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 90)
    # run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 90)
    # run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 180)
    # run_decision_stump(train_data, random.randint(0, 191), random.randint(0, 191), 270)
    print("FINAL")
    pprint(final_ensemble_dict)

    pass
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

def normalize_image_dict_weights(train_data, decision_stump_dict, image_dict):
    increment_percent = 20
    index1 = decision_stump_dict[0]
    index2 = decision_stump_dict[1]
    # total_incorrect = 0
    total = 0.0
    for file in train_data:
        for orient in train_data[file][1]:
            # total += 1
            # Increment the weight for wrongly classified
            if train_data[file][1][orient][index1] <= train_data[file][1][orient][index2]:
                image_dict[file] += image_dict[file]*increment_percent/100

            total += image_dict[file]
    image_dict['TOTAL'] = total
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
    image_dict['TOTAL'] = 0.0
    process_dict = dict()
    process_dict['IN_CLASS'] = 0.0
    process_dict['OUT_CLASS'] = 0.0
    for file in train_data:
        if file not in image_dict:
            image_dict[file] = 1.0
        # for degree in train_data[file][1]:
        total += image_dict[file]
        if train_data[file][1][orient_class][index1] > train_data[file][1][orient_class][index2]:
            process_dict['IN_CLASS'] += image_dict[file]
        else:
            process_dict['OUT_CLASS'] += image_dict[file]
    image_dict['TOTAL'] = total
    # print(process_dict['TRUE']['IN_CLASS'])
    # print(process_dict['TRUE']['OUT_CLASS'])
    # print(process_dict['FALSE']['IN_CLASS'])
    # print(process_dict['FALSE']['OUT_CLASS'])
    print(total)
    error_rate_1 = float(process_dict['OUT_CLASS'])/image_dict['TOTAL']
    error_rate_2 = float(process_dict['IN_CLASS'])/image_dict['TOTAL']
    return error_rate_1, error_rate_2, image_dict
    print(error_rate_1)
    print(error_rate_2)
    print("orient_class:"+str(orient_class))
    # print(error_rate)
    # return error_rate
