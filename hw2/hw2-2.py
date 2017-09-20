import urllib.request
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size
import matplotlib.pyplot as plt

my_url = "https://www.cs.purdue.edu/homes/ribeirob/courses/Fall2017/data/airpollution.csv"

# local_filename, headers = urllib.request.urlretrieve(my_url)
local_filename = "airpollution.csv"
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
    # print("eigenvalues:\n\t{}\n".format(w))
    # print("eigenvectors:\n\t{}\n".format(v))
    w_absolute = np.absolute(w)
    w_absolute_sort = -np.sort(-w_absolute)
    for e_w in w_absolute_sort:
        print(e_w)

    #b
    plt.plot(w_absolute_sort)
    plt.title('Eigenvalues')
    plt.savefig('hw2-2-(b).png')
    # print(matrix_x)
    np_matrix_x = np.array(matrix_x,dtype=float)
    variance_of_matrix_x = np.var(np_matrix_x,axis=0)
    print("variance of different column in original csv:\n\t{}\n".format(variance_of_matrix_x))
    sum_of_variance_of_matrix_x = np.sum(variance_of_matrix_x)
    variance_percantage = 100 * variance_of_matrix_x / sum_of_variance_of_matrix_x
    print("variance percentage:\n\t{}\n".format(variance_percantage))



    #c
    
