import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp


# returns an array of 4 dimensions K X J X N X M
def calculate_posterior_cluster_joint_copy_number(pi, alpha, theta, data):
    K = pi.shape[1]
    J = alpha.shape[1]
    N, M = data.shape[0], data.shape[1]
    result, p = np.zeros((K, J, N, M)), np.zeros((K, J, N, M))
    for n in range(N):
        for m in range(M):
            for j in range(J):
                for k in range(K):
                    p[k, j, n, m] = pi[n, k] + alpha[k, j] + poisson.logpmf(data[n, m], theta[n] * (j + 1))

    for n in range(N):
        for m in range(M):
            result[:, :, n, m] = p[:, :, n, m] - logsumexp(p[:, :, n, m], axis=(0, 1)) # / (np.sum(np.exp(p[:, :, n, m])) + .0000001)

    return np.exp(result)


# returns an array of rates
def update_theta(data, p):
    J = len(p[0, :, 0, 0])
    K = len(p[:, 0, 0, 0])
    N, M = data.shape[0], data.shape[1]
    theta = np.zeros(N)
    denominator = np.zeros((K, J, M))
    for n in range(N):
        for j in range(J):
                denominator[:, j, :] = p[:, j, n, :] * (j + 1)
                theta[n] = np.sum(data[n]) / (np.sum(denominator) + .0000001)
    return theta


# returns a matrix of cluster and copy number K X J
def update_alpha(p):
    return np.sum(p, axis=(2, 3)) / (np.sum(p, axis=(1, 2, 3))[:, np.newaxis] + .0000001)


# returns K
def update_pi(p):
    M = len(p[0, 0, 0, :])
    N = len(p[0, 0, :, 0])
    pi = np.swapaxes(np.sum(p, axis=(1, 3)) / M, 0, 1)
    return pi / (np.sum(pi, axis=1)[:, np.newaxis] + .0000001) #np.sum(p, axis=(1, 2, 3)) / (M * N + .000000001) #


def calculate_transition(p_C_k_m_j):
    J = len(p_C_k_m_j[0, :, 0])
    M = len(p_C_k_m_j[0, 0, :])
    K = len(p_C_k_m_j[:, 0, 0])
    num = np.zeros((K, J, J))
    den = np.zeros((K, J))

    for k in range(K):
        for m in range(1, M):
            j = int(np.argmax(p_C_k_m_j[k, :, m]))
            i = int(np.argmax(p_C_k_m_j[k, :, m-1]))
            num[k, i, j] += 1
            den[k, i] += 1

    trans = num / (den[:, np.newaxis] + .0000000001)

    return trans



def EM(pi, alpha, theta, data, iterations=15):
    K = pi.shape[1]
    J = alpha.shape[1]
    N, M = data.shape[0], data.shape[1]
    p = np.zeros((K, J, N, M))

    # iterations of EM
    for it in range(iterations):

        # E-step:
        p = calculate_posterior_cluster_joint_copy_number(pi, alpha, theta, data)

        # M-step:
        new_pi = update_pi(p)
        new_alpha = update_alpha(p)
        new_theta = update_theta(data, p)

        pi, alpha, theta = new_pi, new_alpha, new_theta

    # Matrix of K X N
    p_Z_n_k = np.sum(p, axis=(1, 3)) / (np.sum(p, axis=(0, 1, 3)) + .0000000001)

    # Matrix of K X J X M
    p_C_k_m_j = np.sum(p, axis=2) / (np.sum(p, axis=(1, 2))[:, np.newaxis, :] + .0000000001)

    trans = calculate_transition(p_C_k_m_j)

    return np.swapaxes(p_Z_n_k, 0, 1), p_C_k_m_j, trans



