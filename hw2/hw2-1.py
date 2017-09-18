import numpy as np

X1000 = np.array([50,10,37,650,400,80,130])
X10 = np.array([20,30,60,10,100,40])

sum_X1000 = np.sum(X1000)
n1000 = X1000.shape[0]
sum_X10 = np.sum(X10)
n10 = X10.shape[0]

N = 500
alpha = 10
beta = 1

sampled_mu1000 = np.random.gamma(shape=(alpha + sum_X1000), scale=1./(beta + n1000), size=N)
sampled_mu10 = np.random.gamma(shape=(alpha + sum_X10), scale=1./(beta + n10), size=N)

greater_tmp = sampled_mu1000 > sampled_mu10
total_1000_ge_10 = np.sum(greater_tmp)

print("Empirical probability P[mu_1000 > mu_10 | Data] =", float(total_1000_ge_10)/N)

#a.1
std_sample_mu1000 = np.std(sampled_mu1000)
std_sample_mu10 = np.std(sampled_mu10)
print("(a)(1)std_sample_mu1000:{}".format(std_sample_mu1000))
print("(a)(1)std_sample_mu10:{}".format(std_sample_mu10))

#a.2
