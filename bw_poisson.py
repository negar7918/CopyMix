from scipy.stats import poisson
import numpy as np
from hmmlearn.base import _BaseHMM, ConvergenceMonitor
from scipy.special import logsumexp
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
import math


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def _logaddexp(a, b):
    if math.isinf(a) and a < 0:
        return b
    elif math.isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))


def _compute_log_xi_sum(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob, log_xi_sum):

    work_buffer = np.zeros((n_components, n_components))
    logprob = logsumexp(fwdlattice[n_samples - 1])

    for t in range(n_samples - 1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[i, j] = (fwdlattice[t, i]
                                     + log_transmat[i, j]
                                     + framelogprob[t + 1, j]
                                     + bwdlattice[t + 1, j]
                                     - logprob)

        for i in range(n_components):
            for j in range(n_components):
                log_xi_sum[i, j] = _logaddexp(log_xi_sum[i, j], work_buffer[i, j])

    return log_xi_sum


def replace_inf(array):
    import sys
    replace = np.isinf(array)
    if (array[replace] < 0).all():
        array[replace] = - sys.maxsize
    elif (array[replace] > 0).all():
        array[replace] = sys.maxsize
    return array


class PoissonHMM(_BaseHMM):

    # Overriding the parent
    def __init__(self, framelogprob, rates, M, *args, **kwargs):
        _BaseHMM.__init__(self, *args, **kwargs)
        # rates for each state
        self.rates = rates
        self.M = M
        self.framelogprob = framelogprob

    def _compute_log_likelihood(self, X):
        J, M, N = self.n_components, X.shape[1], X.shape[0]
        observation_prob = np.zeros((M, J))
        for m in range(M):
            for j in range(J):
                for n in range(N):
                    observation_prob[m, j] += poisson.logpmf(X[n, m], self.rates[j])
        o = observation_prob - logsumexp(observation_prob, axis=0)
        extra_normalized = o - np.amax(o, axis=1)[:, np.newaxis]
        return extra_normalized

    def _initialize_sufficient_statistics(self):
            stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
            stats['post'] = np.zeros(self.n_components)
            stats['obs'] = np.zeros((self.n_components, self.M))
            return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        stats['post'] += posteriors.sum(axis=0)
        for o in obs:
            stats['obs'] += np.multiply(posteriors.T, o)

    def fit(self, X, lengths=None):
        X = check_array(X)
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0

            framelogprob = self._compute_log_likelihood(X)
            logprob, fwdlattice = self._do_forward_pass(framelogprob)
            curr_logprob += logprob
            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors = self._compute_posteriors(fwdlattice, bwdlattice)

            self._accumulate_sufficient_statistics(
                stats, X, framelogprob, posteriors, fwdlattice,
                bwdlattice)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)
            # print(iter)
            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                self.framelogprob = framelogprob
                break

        return self

    def _do_mstep(self, stats):
        super(PoissonHMM, self)._do_mstep(stats)
        denom = stats['post'][:, np.newaxis]
        nom = np.sum(stats['obs'], axis=1)
        self.rates = [a/b for a,b in zip(nom, denom)]


class C:
    def __init__(self, weight_initial, weight_edge, weight_vertex, rates, M):
        self.J = len(weight_initial)
        # The weight of each vertex is an array of size J X M although StubHMM needs M X J
        self.observation_prob = np.swapaxes(weight_vertex, 0, 1)
        # Building HMM
        h = PoissonHMM(n_components=self.J, framelogprob=self.observation_prob, rates=rates, M=M)
        h.transmat_ = weight_edge
        h.startprob_ = weight_initial
        h.framelogprob = self.observation_prob
        self.hmm = h

    def get_hmm(self):
        return self.hmm.startprob_, self.hmm.transmat_, np.exp(np.swapaxes(self.hmm.framelogprob, 0, 1))

    def get_EM_estimation(self, data):
        self.hmm.fit(data)


