
import numpy as np
from matplotlib import use
import math
use('Agg')
import matplotlib.pyplot as plt

p = 0.8
MAX_DEGREE = 102
ECCDF = 1.0
x = []
y = []

for d in xrange(MAX_DEGREE):
	#Be careful with machine precision
	x.append(d)
	y.append(ECCDF)
	ECCDF = ECCDF - (1-p)*p**d

plt.xlim([1,max(x)])
plt.xlabel("node degree", fontsize=18)
plt.ylabel("ECCDF", fontsize=18)
plt.loglog(x,y,"ro")
plt.savefig('ECCDF_plot.pdf')
