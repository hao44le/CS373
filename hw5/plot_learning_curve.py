"""
========================
Plotting Learning Curves
========================

On the left side the learning curve of a naive Bayes classifier is shown for
the digits dataset. Note that the training score and the cross-validation score
are both not very good at the end. However, the shape of the curve can be found
in more complex datasets very often: the training score is very high at the
beginning and decreases and the cross-validation score is very low at the
beginning and increases. On the right side we see the learning curve of an SVM
with RBF kernel. We can see clearly that the training score is still around
the maximum and the validation score could be increased with more training
samples.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

title = "Traning & Validation accuracy"
plt.figure()
plt.title(title)
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.grid()
train_sizes = [100,500,1000,2000,3000]
train_scores_mean = [0.89,0.984,0.998,0.998,0.9993333333333333]
test_scores_mean = [0.940204081632653,0.9817777777777777,0.99575,0.9963333333333333,0.9965]
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Validation accuracy")

plt.legend(loc="best")
plt.savefig("traning&validation accuracy.png")
