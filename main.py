import numpy as np

# Three groups of 50 samples each
# n is number of samples, k is number of groups. k must divide n
n = 150
k = 3
group_size = n // k

X = np.random.normal(0.0, 1.0, (n, 5))
beta = np.transpose(np.array([[1, 2, -5, 3, -3]]))
U = np.random.normal(0.0, 1.0, (k, 1))

eta = np.matmul(X, beta)

# Loop to add random effect U to each group
for i, j in enumerate(U):
    eta[i*group_size:(i+1)*group_size] += j
    print(i, j)

mu = np.exp(eta)
Y = np.random.poisson(mu)

print(z[5]- eta[5])
