import csv
import numpy as np
import datetime
import matplotlib as mpl
mpl.use('Agg')
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size 
import matplotlib.pyplot as plt

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
country_expense_dict = dict()

#3
headers = []
all_rows = []

for row in spamreader:
  total_number_of_rows += 1
  if total_number_of_rows == 1:
    #header
    headers = row
    continue
  invoice_no,stock_code,description,quantity,invoice_date,unit_price,customer_id,country = row
  #1.b
  stock_code_set.add(stock_code)
  if stock_code == '20685' : unit_cost_array_for_20685.append(unit_price)
  
  #2.b
  date = datetime.datetime.strptime(invoice_date, "%m/%d/%y %H:%M")
  hour_in_date = date.hour
  if hour_in_date in hour_sales_dict:
    hour_sales_dict[hour_in_date] += int(quantity)
  else:
    hour_sales_dict[hour_in_date] = int(quantity)
  
  #2.c
  total = float(unit_price) * int(quantity)
  if country in country_expense_dict:
    country_expense_dict[country] += total
  else:
    country_expense_dict[country] = total
  
  #3
  all_rows.append(row)

#1.a
total_number_of_rows -= 1

#2.c
#we want to plot country with more than $50,000 spending
filtered_country_expense_dict = dict()
for country in country_expense_dict:
  total_speding = country_expense_dict[country]
  if total_speding > 50000:
    filtered_country_expense_dict[country] = total_speding


#3
np_all_rows = np.array(all_rows)
np.random.shuffle(np_all_rows)
split_at_row = len(np_all_rows) / 2
first_half_rows = np_all_rows[:split_at_row]
second_half_rows = np_all_rows[split_at_row:]

#Print for pdf
print("1")
print("\t(a) total_number_of_rows is " + str(total_number_of_rows))

print("\t(b) number_of_unique_items based on unique stock code are " + str(len(stock_code_set)))

print("2")
np_array = np.array(unit_cost_array_for_20685).astype(np.float)
unit_cost_array_20685 = np.mean(np_array)
print("\t(a) average unit price for the product with stock code 20685 is {0}".format(unit_cost_array_20685))

max_hour = max(hour_sales_dict, key=hour_sales_dict.get)
max_quantity_at_max_hour = hour_sales_dict[max_hour]
print("\t(b) hour in the day are most items sold in the given data set is {0} and the maximum quantity is {1}".format(max_hour,max_quantity_at_max_hour))


print("\t(c) generate bar graph using {0}".format(filtered_country_expense_dict))
plt.bar(range(len(filtered_country_expense_dict)), filtered_country_expense_dict.values(), align='center')
plt.xticks(range(len(filtered_country_expense_dict)), list(filtered_country_expense_dict.keys()))
plt.grid(True)
plt.xlabel('Country')
plt.ylabel('Total Spent $')
plt.title('The amount spent by residents of each country > $50,000')
plt.savefig('hw1-2.c.png')


print("3")
print("\tgenerate csvs")
output_1 = "output-1.csv"
output_1_file = open(output_1,"wb")
output_1_file.write(",".join(headers))
output_1_file.write("\n")
for x,row in enumerate(first_half_rows):
  output_1_file.write(",".join(row))
  if x != len(first_half_rows) - 1:
    output_1_file.write("\n")
output_1_file.close()

output_2 = "output-2.csv"
output_2_file = open(output_2,"wb")
output_2_file.write(",".join(headers))
output_2_file.write("\n")
for x,row in enumerate(second_half_rows):
  output_2_file.write(",".join(row))
  if x != len(second_half_rows) - 1:
    output_2_file.write("\n")
output_2_file.close()
csvfile.close()
