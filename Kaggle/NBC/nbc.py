from sklearn.naive_bayes import GaussianNB
import numpy as np

training_file_name = "Loan_Training.csv"
testing_file_name = "Loan_ToPredict.csv"

raw_data = []
raw_target = []
import csv
line_number = 0
info = csv.reader(open(training_file_name), skipinitialspace=True)
for row in info:
    line_number += 1
    if line_number == 1:
        continue
    raw_data.append(row[1:-1])
    raw_target.append(row[-1])
# 2

np_data = np.array(raw_data)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(len(raw_data[0])):
    np_data[:, i] = le.fit_transform(np_data[:, i])
np_data = np.array(np_data, dtype=float)

np_targets = np.array(raw_target)
gnb = GaussianNB()
gnb.fit(np_data, np_targets)
#
test_data = []
raw_output = []
line_number = 0
info = csv.reader(open(training_file_name), skipinitialspace=True)
for row in info:
    line_number += 1
    if line_number == 1:
        continue
    test_data.append(row[1:-1])
    raw_output.append(row)

test_data = np.array(test_data)
for i in range(len(test_data[0])):
    test_data[:, i] = le.fit_transform(test_data[:, i])
test_data = np.array(test_data, dtype=float)
y_pred = gnb.predict(test_data)

reader = csv.reader(open(testing_file_name), skipinitialspace=True)
writer = csv.writer(open('out.csv', 'w'))
line_number = 0
for row in reader:
    line_number += 1
    if line_number == 1:
        writer.writerow([row[0], row[-1]])
        continue
    o_row = [row[0]]
    o_row.append(y_pred[line_number - 2])
    print(o_row)
    writer.writerow(o_row)
