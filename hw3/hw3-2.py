import numpy as np
import sys

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
print("training_file_name {}".format(training_file_name))
print("testing_file_name {}".format(testing_file_name))
line_number = 0

#1.d
goodForGroups_set,city_set,state_set,stars_set,is_open_set,alcohol_set,noiseLevel_set,attire_set,priceRange_set,delivery_set,waiterService_set,smoking_set,outdoorSeating_set,caters_set,goodForKids_set = set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set()
sets = [goodForGroups_set,city_set,state_set,stars_set,is_open_set,alcohol_set,noiseLevel_set,attire_set,priceRange_set,delivery_set,waiterService_set,smoking_set,outdoorSeating_set,caters_set,goodForKids_set]
set_names = ["goodForGroups_set","city_set","state_set","stars_set","is_open_set","alcohol_set","noiseLevel_set","attire_set","priceRange_set","delivery_set","waiterService_set","smoking_set","outdoorSeating_set","caters_set","goodForKids_set"]

#1.g
numerator_factor = 1
priceRange_dict = {'':[0,0],'1':[0,0],'2':[0,0],'3':[0,0],'4':[0,0]}
alcohol_dict = {'none':[0,0],'full_bar':[0,0],'beer_and_wine':[0,0]}
noiseLevel_dict = {'':[0,0],'very_loud':[0,0],'loud':[0,0],'quiet':[0,0],'average':[0,0]}
attire_dict = {'':[0,0],'dressy':[0,0],'casual':[0,0],'formal':[0,0]}
all_dict = {'priceRange_dict':priceRange_dict,'alcohol_dict':alcohol_dict,'noiseLevel_dict':noiseLevel_dict,'attire_dict':attire_dict}

#2
raw_data = []

with open(training_file_name) as in_file:
    for line in in_file.readlines():
        line_number += 1
        line = line.strip()
        if line_number == 1:
            #header.skip
            continue
        else:
            [goodForGroups,city,state,_,_,stars,_,_,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids] = line.split(",")
            #1.d
            goodForGroups_set.add(goodForGroups)
            city_set.add(city)
            state_set.add(state)
            stars_set.add(stars)
            is_open_set.add(is_open)
            alcohol_set.add(alcohol)
            noiseLevel_set.add(noiseLevel)
            attire_set.add(attire)
            priceRange_set.add(priceRange)
            delivery_set.add(delivery)
            waiterService_set.add(waiterService)
            smoking_set.add(smoking)
            outdoorSeating_set.add(outdoorSeating)
            caters_set.add(caters)
            goodForKids_set.add(goodForKids)

            #1.g
            if goodForGroups == '0':
                priceRange_dict[priceRange] = [priceRange_dict[priceRange][0]+1,priceRange_dict[priceRange][1]]
                alcohol_dict[alcohol] = [alcohol_dict[alcohol][0]+1,alcohol_dict[alcohol][1]]
                noiseLevel_dict[noiseLevel] = [noiseLevel_dict[noiseLevel][0]+1,noiseLevel_dict[noiseLevel][1]]
                attire_dict[attire] = [attire_dict[attire][0]+1,attire_dict[attire][1]]
            else:
                priceRange_dict[priceRange] = [priceRange_dict[priceRange][0],priceRange_dict[priceRange][1]+1]
                alcohol_dict[alcohol] = [alcohol_dict[alcohol][0],alcohol_dict[alcohol][1]+1]
                noiseLevel_dict[noiseLevel] = [noiseLevel_dict[noiseLevel][0],noiseLevel_dict[noiseLevel][1]+1]
                attire_dict[attire] = [attire_dict[attire][0],attire_dict[attire][1]+1]

            #2
            raw_data.append([goodForGroups,city,state,stars,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids])

#1.d
# total_number_of_distinct_values = 0
# for index,m_set in enumerate(sets):
    # print("{} {}:{}".format(set_names[index],len(m_set),m_set))
    # total_number_of_distinct_values += len(m_set)
# print("total_number_of_distinct_values:{}".format(total_number_of_distinct_values))

#1.g
# for key in all_dict:
#     print("{}:{}".format(key,all_dict[key]))

#2
np_array = np.array(raw_data)
X_train = np_array[:, 1:]
d = X_train.shape[1]
n = X_train.shape[0]

#Laplace smoothing
numerator_factor = 1

# calculate the class distirbution
class_probs = {}
y_train = np_array[:,0]
classes = np.unique(y_train)
for c in classes:
    class_probs[c] = (np.array(y_train == c).sum() + numerator_factor) / (n + len(classes))

# calculate the feature probabilities given the calsses
possible_values = [set(X_train[:, feature]) for feature in range(d)]
print(possible_values[0])
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
print(feature_probs[0,'1']['NELLIS AFB'])
print(feature_probs[0,'0']['NELLIS AFB'])
#Read testing files
line_number = 0
testing_data = []
with open(testing_file_name) as in_file:
    for line in in_file.readlines():
        line_number += 1
        line = line.strip()
        if line_number == 1:
            #header.skip
            continue
        else:
            [goodForGroups,city,state,_,_,stars,_,_,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids] = line.split(",")
            testing_data.append([goodForGroups,city,state,stars,is_open,alcohol,noiseLevel,attire,priceRange,delivery,waiterService,smoking,outdoorSeating,caters,goodForKids])
testing_np_array = np.array(testing_data)
X_test = testing_np_array[:,1:]
y_test = testing_np_array[:,0]
print("X_test:{}".format(X_test.shape))
print("y_test:{}".format(y_test.shape))
#testing
test_size = len(X_test)
y_pred = []
for i in range(test_size):
    posterior_prob = {c: 0 for c in classes}
    y_max = classes[0]

    for c in classes:
        # Compute the posterior prob. for class c
        posterior_prob[c] = class_probs[c]
        for j in range(d):
            x = X_test[i, j]
            posterior_prob[c] *= feature_probs[j, c][x]

        # Update which class has the max posterior
        if posterior_prob[c] >= posterior_prob[y_max]:
            y_max = c
    y_pred.append(y_max)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
