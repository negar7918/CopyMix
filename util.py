import numpy as np
from hmmlearn import hmm
from scipy import stats


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
        rates = self.rates * add_noise(state)
        return [stats.poisson(rate).rvs() for rate in rates]


def generate_hmm(rate, num_of_states, trans, start_prob, num_of_samples):
    model = PoissonHMM(rate, n_components=num_of_states)
    model.startprob_ = np.array(start_prob)
    model.transmat_ = np.array(trans)
    Y, C = model.sample(num_of_samples)
    print("Y values:")
    print(Y)
    print("C values:")
    print(C)
    return C, Y


def generate_Z(prob, num_of_cells):
    Z = []
    result = np.random.multinomial(1, prob, num_of_cells)
    for element in result:
        Z.append(int(np.where(element == 1) + np.ones(1)))
    print("Z values:")
    print(Z)
    return Z