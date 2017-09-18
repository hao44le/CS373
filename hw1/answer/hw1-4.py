import numpy
from numpy.linalg import inv

#Q4 (a) code
rows = 100
columns = 20
x = numpy.zeros(shape=(rows,columns))

for i in range(0,rows):
  for j in range(0,columns):
    if i <= j:
      x[i][j] = 2 * i + j * j + 1
    else:
      x[i][j] = i * i - 2 * j

vector_shape = 100
y = numpy.zeros(shape=(vector_shape))
for i in range(0,vector_shape):
  y[i] = i * i - 1

x_transpose = numpy.transpose(x)

left_b = inv((numpy.dot(x_transpose,x)))
right_b = numpy.dot(x_transpose,y)
b = numpy.dot(left_b,right_b)
print("Q4 (a) b: {0}".format(b))

#Code for Q4 b
inner_product = numpy.inner(x[0],b)
print("Q4 (b) inner_product:{0}".format(inner_product))
