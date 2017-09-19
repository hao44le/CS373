import urllib.request
import numpy as np

my_url = "https://www.cs.purdue.edu/homes/ribeirob/courses/Fall2017/data/airpollution.csv"

local_filename, headers = urllib.request.urlretrieve(my_url)

with open(local_filename) as in_file:
    number_of_lines = 0
    matrix_x = []

    #to compute the matrix_x
    for line in in_file.readlines():
        number_of_lines += 1
        if number_of_lines == 1:
            continue
        line = line.strip()
        # (city,so2,temp,manu,popul,wind,percip,predays) = line.split(",")
        # matrix_x.append([temp,manu,popul,wind,percip,predays])
        matrix_x.append(line.split(",")[-6:])

    #to compute the matrix_s
    matrix_s = np.zeros((6,6))
    for x_j in matrix_x:
        x_j = np.array([x_j],dtype=float)
        x_j_transpose = np.transpose(x_j)
        dot_product = np.multiply(x_j,x_j_transpose)
        matrix_s = np.add(matrix_s,dot_product)

    #a
    w, v = np.linalg.eig(matrix_s)
    w_absolute = np.absolute(w)
    w_absolute_sort = -np.sort(-w_absolute)
    for e_w in w_absolute_sort:
        print(e_w)

    #b
    
