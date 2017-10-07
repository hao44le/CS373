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

def report_stat(y_pred,y_test):
    test_size = len(y_test)
    #Calculate stats
    zero_one_loss = (np.array(y_pred != y_test).sum())/(test_size)
    # print("ZERO-ONE LOSS={:.50f}".format(zero_one_loss))
    return zero_one_loss

training_file_name = "yelp2_train.csv"
raw_data = read_data(training_file_name)
raw_data = np.array(raw_data)

per_arr = [0.01,0.1,0.5]
for percent in per_arr:
    print(percent)
    zero_one_loss_arr = np.zeros(10,)
    for i in range(0,10):
        # print("\t{}".format(i))
        np.random.shuffle(raw_data)
        training_percent = int(len(raw_data) * percent)
        training_data, testing_data = raw_data[:training_percent], raw_data[training_percent:]

        testing_np_array = np.array(testing_data)
        y_test = testing_np_array[:,0]

        #get majority y_pred
        # print(max_class)
        y_pred = []
        for x in y_test:
            y_pred.append('1')
        y_pred = np.array(y_pred)
        zero_one_loss = report_stat(y_pred,y_test)
        zero_one_loss_arr[i] = zero_one_loss
    print("mean of zero_one_loss_arr:{}".format(np.mean(zero_one_loss_arr)))
