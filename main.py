import numpy as np
import statsmodels.api as sm


def mse(a, b):
    return (np.square(a-b)).mean(axis=None)


if __name__ == "__main__":
    np.random.seed(seed=42)
    # Three groups of 50 samples each
    # n is number of samples, k is number of groups. k must divide n
    n = 150
    k = 3
    group_size = n // k

    X = np.random.normal(0.0, 1.0, (n, 5))
    beta = np.transpose(np.array([[1, 2, -5, 3, -3]]))
    U = np.random.normal(0.0, 1, (k, 1))

    eta = np.matmul(X, beta)

    # Loop to add random effect U to each group
    for i, j in enumerate(U):
        eta[i*group_size:(i+1)*group_size] += j

    mu = np.exp(eta)
    Y = np.random.poisson(mu)

    # Now we estimate beta from the observations assuming independence

    poisson_glm = sm.GLM(Y, X, family=sm.families.Poisson())
    glm_results = poisson_glm.fit()

    # make beta_hat a 2-d array
    beta_hat = glm_results.params[:, np.newaxis]
    print(mse(beta_hat, beta))

