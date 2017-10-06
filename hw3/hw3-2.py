import numpy as np
import sys

def read_data(filename):
    raw_data = []
    line_number = 0
    with open(filename) as in_file:
        for line in in_file.readlines():
            line_number += 1
            line = line.strip()
            if line_number == 1:
                #header.skip
                continue
            else:
                [goodForGroups,city,state,_,_,stars,_,_,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids] = line.split(",")
                raw_data.append([goodForGroups,city,state,stars,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids])
    return raw_data

#Laplace smoothing
numerator_factor = 1.0

def training(m_raw_data):
    np_array = np.array(m_raw_data)
    X_train = np_array[:, 1:]
    d = X_train.shape[1]
    n = X_train.shape[0]

    # calculate the class distirbution
    class_probs = {}
    y_train = np_array[:,0]
    classes = np.unique(y_train)
    for c in classes:
        class_probs[c] = (np.array(y_train == c).sum() + numerator_factor) / (n + len(classes))

    # calculate the feature probabilities given the calsses
    possible_values = [set(X_train[:, feature]) for feature in range(d)]

    feature_probs = {(j, c): {v: 0 for v in possible_values[j]}
                     for c in classes for j in range(d)}
    for j in range(d):
        for c in classes:
            # This gives us the j-th feature of instances in class c
            in_class_c = X_train[y_train == c, j]
            for x in possible_values[j]:
                numerator = sum(in_class_c == x) + numerator_factor
                denominator = len(in_class_c) + len(possible_values[j])
                feature_probs[j, c][x] = numerator / denominator
    return class_probs,feature_probs,classes,d,possible_values


def testing(m_testing_data,classes,class_probs,feature_probs,possible_values):
    #Read testing files
    testing_np_array = np.array(m_testing_data)
    X_test = testing_np_array[:,1:]
    y_test = testing_np_array[:,0]

    #testing
    test_size = len(X_test)
    y_pred = []
    y_pred_pro = []
    for i in range(test_size):
        posterior_prob = {c: 0 for c in classes}
        y_max = classes[0]

        for c in classes:
            # Compute the posterior prob. for class c
            posterior_prob[c] = class_probs[c]
            for j in range(d):
                x = X_test[i, j]
                if x not in feature_probs[j, c]:
                    # Smoothing for unseen features
                    posterior_prob[c] *= (numerator_factor / len(possible_values[j]))
                else:
                    posterior_prob[c] *= feature_probs[j, c][x]

            # Update which class has the max posterior
            if posterior_prob[c] >= posterior_prob[y_max]:
                y_max = c
        y_pred.append(y_max)
        sum_prob = 0.0
        for v in posterior_prob:
            sum_prob += posterior_prob[v]
        y_pred_pro.append(posterior_prob[y_max]/sum_prob)
    return y_pred,y_pred_pro,y_test

def report_stat(y_pred,y_test,y_pred_pro):
    test_size = len(y_test)
    #Calculate stats
    zero_one_loss = (np.array(y_pred != y_test).sum())/(test_size)
    print("ZERO-ONE LOSS={}".format(zero_one_loss))
    sum_of_prob = 0.0
    for pro in y_pred_pro:
        sum_of_prob += (1-pro)*(1-pro)
    squard_loss = (sum_of_prob)/(test_size)
    print("SQUARED LOSS={}".format(squard_loss))

training_file_name = sys.argv[1]
raw_data = read_data(training_file_name)
# without random split
class_probs,feature_probs,classes,d,possible_values = training(raw_data)

#Read testing files
testing_data = read_data(sys.argv[2])

y_pred,y_pred_pro,y_test = testing(testing_data,classes,class_probs,feature_probs,possible_values)
report_stat(y_pred,y_test,y_pred_pro)
