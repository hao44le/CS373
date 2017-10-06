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
    feature = row[1:6] #Amount Requested	Amount Funded By Investors	Interest Rate	Loan Length	CREDIT Grade
    feature.append(row[8])#Monthly PAYMENT
    feature.append(row[9])#Total Amount Funded
    feature.append(row[10])#Debt-To-Income Ratio
    feature.append(row[14])#Monthly Income
    raw_data.append(feature)
    print(feature)
    raw_target.append(row[-1])
# 2
print(len(raw_data))
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
    feature = row[1:6] #Amount Requested	Amount Funded By Investors	Interest Rate	Loan Length	CREDIT Grade
    feature.append(row[8])#Monthly PAYMENT
    feature.append(row[9])#Total Amount Funded
    feature.append(row[10])#Debt-To-Income Ratio
    feature.append(row[14])#Monthly Income
    test_data.append(feature)
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
