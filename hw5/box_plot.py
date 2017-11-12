import numpy as np
import matplotlib as mpl
a = [ 0.882,0.882,0.88,0.88,0.878,0.878,0.87933333,0.88066667,0.88133333,0.88133333,0.88066667,0.88066667,0.88133333,0.88,0.88133333,0.88066667,0.88066667,0.88066667,0.88,0.882]
b = [ 0.99666667,0.99733333,0.99733333,0.99733333,0.99733333,0.99666667,0.99733333,0.99666667,0.99666667,0.99666667,0.99666667,0.99666667,0.99666667,0.99733333,0.99733333,0.99733333,0.99666667,0.99666667,0.99666667,0.99666667]

# a = np.random.rand(10)
# b = np.random.rand(10)
## agg backend is used to create plot as a .png file
mpl.use('agg')

import matplotlib.pyplot as plt
data = [a,b]
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data)

plt.xticks([1, 2], ['a', 'b'])

# Save the figure
fig.savefig('boxplot.png')
