import numpy as np
from hmmlearn import hmm, utils
from scipy import stats
import math

#rng = np.random.default_rng(12)

def add_noise(state):
    if state == 0:
        state = .001
    return state


class PoissonHMM(hmm._BaseHMM):

    # Overriding the parent
    def __init__(self, rates, *args, **kwargs):
        hmm._BaseHMM.__init__(self, *args, **kwargs)
        self.rates = rates

    # Overriding the parent
    def _generate_sample_from_state(self, state, random_state=None):
        rates = self.rates #* add_noise(state)
        return [stats.poisson(rate).rvs() for rate in rates]


class NormalHMM(hmm._BaseHMM):

    # Overriding the parent
    def __init__(self, r, means, var, *args, **kwargs):
        hmm._BaseHMM.__init__(self, *args, **kwargs)
        self.means = means
        self.var = var
        self.random_state = r

    # Overriding the parent
    def _generate_sample_from_state(self, state, random_state=None):
        means = self.means * add_noise(state)
        return [self.random_state.normal(loc=mean, scale=math.sqrt(self.var), size=None) for mean in means]


def generate_hmm(rate, num_of_states, trans, start_prob, num_of_samples):
    model = PoissonHMM(rate, n_components=num_of_states)
    model.startprob_ = np.array(start_prob)
    model.transmat_ = np.array(trans)
    Y, C = model.sample(num_of_samples)
    M = Y.shape[0]
    prob_Cm = np.zeros((M, num_of_states))
    for m in range(M):
        if C[m] == 0:  # state 0
            prob_Cm[m, 0] = 1
        else:  # state 1
            prob_Cm[m, 1] = 1
    return C, Y, prob_Cm


def pdf(point, mean, var):
    return 1 / (math.sqrt(math.pi * 2) * math.sqrt(var)) * math.exp(-.5 * ((point-mean)**2) / var)


def generate_hmm_normal(means, var, num_of_states, trans, start_prob, num_of_samples, r):
    model = NormalHMM(r, means, var, n_components=num_of_states)
    model.startprob_ = np.array(start_prob)
    model.transmat_ = np.array(trans)
    Y, C = model.sample(num_of_samples)
    M = Y.shape[0]
    prob_Cm = np.zeros((M, num_of_states))
    for m in range(M):
        if C[m] == 2:
            prob_Cm[m, 2] = 1
        elif C[m] == 5:
            prob_Cm[m, 5] = 1
    return C, Y, prob_Cm


def generate_Z(prob, num_of_cells, r):
    Z = []
    result = r.multinomial(1, prob, num_of_cells)
    for element in result:
        Z.append(int(np.where(element == 1) + np.ones(1)))
    print("Z values:")
    print(Z)
    return Z


class StubHMM(hmm._BaseHMM):
    def __init__(self, emiss, *args, **kwargs):
        hmm._BaseHMM.__init__(self, *args, **kwargs)
        self.emiss = emiss

    def _compute_log_likelihood(self, X):
        return self.emiss


def calculate_most_probable_states(cells, trans, emiss, weight_initial, p):
    M = len(emiss[0, 0, :])
    J = len(emiss[0, :, 0])
    K = len(emiss[:, 0, 0])
    N = len(cells[:, 0])
    states = np.zeros((K, M))

    for k in range(K):
        indexes = np.where(p[:, k] > (1/K))[0]
        cells_in_cluster = cells[indexes]
        model = StubHMM(n_components=J, emiss=np.swapaxes(emiss[k], 0, 1))
        utils.normalize(trans[k], axis=1)
        model.transmat_ = trans[k]
        model.startprob_ = weight_initial[k] / np.sum(weight_initial[k])
        a = np.array(cells_in_cluster).flatten()[:, np.newaxis]
        l = np.ones(len(cells_in_cluster), dtype=int) * M
        if len(cells_in_cluster) > 0:
            l_probs, s = model.decode(a, lengths=l, algorithm="viterbi")
            states[k] = np.mean(s.reshape(len(cells_in_cluster), M), axis=0)
    return states