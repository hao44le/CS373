import csv
import numpy as np
import datetime

csv_name = 'retail.csv'
csvfile = open(csv_name,'rb')
spamreader = csv.reader(csvfile)

#1.a
total_number_of_rows = 0

#1.b
stock_code_set = set()

#2.a
unit_cost_array_for_20685 = []

#2.b
hour_sales_dict = dict()

#2.c


#3
for row in spamreader:
  total_number_of_rows += 1
  if total_number_of_rows == 1:
    #header
    continue
  invoice_no,stock_code,description,quantity,invoice_date,unit_price,customer_id,country = row
  stock_code_set.add(stock_code)
  if stock_code == '20685' : unit_cost_array_for_20685.append(unit_price)
  date = datetime.datetime.strptime(invoice_date, "%m/%d/%y %H:%M")
  hour_in_date = date.hour
  hour_sales_dict[hour_in_date] = int(quantity)
total_number_of_rows -= 1

print("1")
print("\t(a)total_number_of_rows is " + str(total_number_of_rows))

print("\t(b)number_of_unique_items are " + str(len(stock_code_set)))

print("2")
np_array = np.array(unit_cost_array_for_20685).astype(np.float)
unit_cost_array_20685 = np.mean(np_array)
print("\t(a)average unit price for the product with stock code 20685 is {0}".format(unit_cost_array_20685))

print("\t(b)hour in the day are most items sold in the given data set is {0}".format(hour_sales_dict))

csvfile.close()
