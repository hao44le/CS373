import numpy

rows = 100
columns = 20
x = numpy.zeros(shape=(rows,columns))

print("Q4 a)")
print("---------------------------------")
for i in range(0,rows):
  for j in range(0,columns):
    if i <= j:
      x[i][j] = 2 * i + j * j + 1
    else:
      x[i][j] = i * i - 2 * j
print("\t\tx shape:{0}".format(x.shape))

vector_shape = 100
y = numpy.zeros(shape=(vector_shape))
for i in range(0,vector_shape):
  y[i] = i * i - 1
print("\t\ty.shape:{0}".format(y.shape))

x_transpose = numpy.transpose(x)
print("\t\tx_transpose shape:{0}".format(x_transpose.shape))

left_b = 1 / (numpy.dot(x_transpose,x))
print("\t\tleft_b shape:{0}".format(left_b.shape))
right_b = numpy.dot(x_transpose,y)
print("\t\tright b shape:{0}".format(right_b.shape))

b = numpy.dot(left_b,right_b)
print("\t\t{0}".format(b))
print("\t\tb shape :{0}".format(b.shape))

print("---------------------------------")
print("Q4 b)")
inner_product = numpy.inner(x[0],b)
print("\t\tx[0]:{0}".format(x[0]))
print("\t\tb:{0}".format(b))
print("\t\tinner_product:{0}".format(inner_product))
