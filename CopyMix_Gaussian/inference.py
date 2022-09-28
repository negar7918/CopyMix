from scipy.stats import dirichlet, multinomial
import numpy as np
import math
from hmmlearn.base import _BaseHMM
from hmmlearn import utils, hmm
from scipy.special import digamma, logsumexp
import tqdm


class Dirichlet:
    def __init__(self, deltas):
        self.deltas = deltas

    def get_distribution(self):
        return dirichlet(self.deltas)

    def get_expectation(self, index):
        return self.deltas[index] / np.sum(self.deltas)

    def get_expectation_of_log(self):
        return digamma(self.deltas) - digamma(np.sum(self.deltas))


class Categorical:
    def __init__(self, p):
        self.n = 1
        self.p = p

    def get_distribution(self):
        return multinomial(self.n, self.p)

    def get_expectation_of_log_of_probability(self):
        # E[log(∏k=1 p_k ^ I(Z_n=k))] = ∑k=1 ∑n=1 E[I(Z_n=k)] * log(p_nk) = ∑k=1 ∑n=1 p_nk * log(p_nk)
        l = list(self.p)
        i = 0
        for element in l:
            if element == 0:
                l[i] = 0.000000001
            i += 1
        return np.sum(self.p * np.log(l))


class Pi(Dirichlet):
    def __init__(self, *args, **kwargs):
        Dirichlet.__init__(self, *args, **kwargs)


class Z(Categorical):
    def __init__(self, *args, **kwargs):
        Categorical.__init__(self, *args, **kwargs)

    def get_expectation(self, k):
        distribution = super(Z, self).get_distribution()
        return distribution.pmf(k)


class A(Dirichlet):
    def __init__(self, *args, **kwargs):
        Dirichlet.__init__(self, *args, **kwargs)


class Gamma:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def get_expectation(self):
        return self.alpha / self.beta

    def get_expectation_of_log(self):
        return digamma(self.alpha) - math.log(self.beta)


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.eps = sigma

    def get_distribution(self):
        return np.random.normal(loc=self.mu, scale=math.sqrt(self.eps), size=None)

    def get_expectation_mu(self):
        return self.mu

    def get_expectation_mu_square(self):
        return (self.mu ** 2) + self.eps


class StubHMM(_BaseHMM):
    def __init__(self, *args, **kwargs):
        hmm._BaseHMM.__init__(self, *args, **kwargs)

    # An HMM with hardcoded observation probabilities
    # def _compute_log_likelihood(self, X):
    #     return self.framelogprob


# A Directed Graph
class C:
    def __init__(self, M, weight_initial, weight_edge, weight_vertex):
        self.J = len(weight_initial)
        self.M = M
        # The weight of each vertex is an array of size J X M although StubHMM needs M X J
        self.observation_prob = np.swapaxes(weight_vertex, 0, 1)
        # Building HMM
        h = StubHMM(self.J)
        h.transmat_ = weight_edge + .0000000000001
        h.startprob_ = weight_initial + .0000000000001
        self.hmm = h
        # Calculating forward and backward probabilities
        logprob, fwdlattice = self.hmm._do_forward_pass(self.observation_prob)
        bwdlattice = self.hmm._do_backward_pass(self.observation_prob)
        self.fwd = replace_inf(fwdlattice)
        self.bwd = replace_inf(bwdlattice)

    # Returns an array of probabilities, J X M
    def get_expectation(self):
        # HMM: fwd * bwd / prob(observations)
        # Directed Graph: fwd * bwd
        q = self.fwd + self.bwd
        # log normalization
        utils.log_normalize(q, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(np.swapaxes(q, 0, 1))

    # Returns an array of probabilities, J X J x M-1
    # E(I(C_km=j, C_km-1))
    def expectation_two(self):
        state_probability = np.zeros((self.J, self.J, self.M - 1))
        # HMM: fwd * emission * transition * bwd / prob(observations)
        # Directed Graph: fwd_{m-1} * transition * emission_{m} * bwd_{m}

        for i in range(self.J):
            for j in range(self.J):
                for m in range(1, self.M):
                    state_probability[i, j, m - 1] = self.fwd[m - 1, j] + np.log(self.hmm.transmat_[i, j]) + \
                                                     self.observation_prob[m, j] + self.bwd[m, j]

        result = np.exp(state_probability - logsumexp(state_probability, axis=(0, 1)))  # log normalization

        return result

    # Returns an array of probabilities, J X J
    # ∑m=2 E(I(C_km=j, C_km-1))
    def sum_of_expectation_two(self):
        return np.sum(self.expectation_two(), axis=2)

    # ∑m=2 ∑i=1 ∑j=1 E(I(C_km=j, C_km-1=i)) log { E(I(C_km=j, C_km-1=i)) / E(I(C_km-1=i)) }
    def get_expectation_of_log_of_probability(self):
        expected_hidden = self.get_expectation()
        expected_hidden_two = self.expectation_two()
        sum = 0
        for i in range(self.J):
            for j in range(self.J):
                for m in range(1, self.M):
                    sum += expected_hidden_two[i, j, m-1] * math.log(expected_hidden_two[i, j, m-1] / (expected_hidden[j, m-1] + .00000001) + .00000001)
        return sum


# Returns an array of updated delta
def update_delta(delta, pi_prev):
    sum_of_expectations = [np.sum(pi_prev[:, k]) for k in range(len(pi_prev[0]))]
    return delta + sum_of_expectations


# Returns an array of updated lambda
def update_lambda(lam, sum_of_expectation_two):
    return lam + sum_of_expectation_two


# Returns the updated beta param of the gamma
def update_beta_gam(params):
    beta, pi_prev, expected_hidden, theta_prev, tau_prev, cells = params
    sums = 0
    J = len(expected_hidden[0, :, 0])
    K = len(expected_hidden[:, 0, 0])
    N = cells.shape[0]
    for n in range(N):
        for k in range(K):
            for j in range(J):
                val = (cells[n,:] ** 2) - 2 * cells[n,:] * j * theta_prev[n] + (j ** 2) * (tau_prev[n] + theta_prev[n] ** 2)
                val_summed_over_m = np.sum(np.multiply(expected_hidden[k, j, :], val))
                sums += pi_prev[n, k] * val_summed_over_m
    return beta + .5 * sums


# Returns the updated alpha param of the gamma
def update_alpha_gam(alpha_gam, y):
    N, M = y.shape[0], y.shape[1]
    return alpha_gam + N * M / 2


def update_pi_inner(params):
    K, J, M, theta, tau, expected_hidden, alpha, beta, cell = params
    sums = np.zeros((K))
    val = 0
    for k in range(K):
        for j in range(J):
            for m in range(M):
                prob_c = expected_hidden[k, j, m]
                expectation = calculate_expectation_of_D(j, theta, tau, alpha, beta, cell[m])
                val += prob_c * expectation
        sums[k] = val
        val = 0
    return sums


# Returns an array of updated pi
def update_pi(delta, expected_hidden, theta, tau, alpha, beta, cells):
    K = delta.size
    J = len(expected_hidden[0, :, 0])
    M = len(expected_hidden[0, 0, :])
    N = len(cells)
    log_normalized = np.zeros((N,K))
    pi = Pi(delta)

    import multiprocessing as mp
    pool = mp.Pool(1)
    sums = pool.map(update_pi_inner, [(K, J, M, theta[n], tau[n], expected_hidden, alpha, beta, cells[n]) for n in range(N)])
    pool.close()

    updated_pi = pi.get_expectation_of_log() + sums

    for n in range(N):
        log_normalized[n, :] = updated_pi[n] - logsumexp(updated_pi[n])

    return np.exp(log_normalized)


# input: J, J x M, start position of each chromosome
def correct_for_chrom(weight_vertex, locs):
    for i in locs[:-1]:
        prev_i = int(i)
        weight_vertex[:, prev_i + 1] = weight_vertex[:, 0]

    return weight_vertex


def replace_inf(array):
    import sys
    replace = np.isinf(array)
    if (array[replace] < 0).all():
        array[replace] = - sys.maxsize
    elif (array[replace] > 0).all():
        array[replace] = sys.maxsize
    return array


# Returns an array of M: expected value of D at j, n for all m in M
def calculate_expectation_of_D(j, theta_n, tau_n, alpha, beta, cell_nm):
    state = j
    norm = Gaussian(theta_n, tau_n)
    gam = Gamma(alpha, beta)
    E_sigma_log_1sigma2 = gam.get_expectation_of_log()
    E_sigma_1sigma2 = gam.get_expectation()
    E_mu_n = norm.get_expectation_mu()
    E_mu2_n = norm.get_expectation_mu_square()
    result = - .5 * (math.log(2* math.pi) - E_sigma_log_1sigma2 +
                     (cell_nm ** 2) * E_sigma_1sigma2 - 2 * cell_nm * state * E_mu_n * E_sigma_1sigma2 +
                     ((state ** 2) * E_mu2_n * E_sigma_1sigma2))
    return result


# Returns an array of weights, K X J X J
# No normalization needed since these are weights and not probabilities
def update_weight_edge(lam, pi):
    K = len(pi[0])
    J = len(lam[0, 0, :])
    weight_edge = np.zeros((K, J, J))
    for i in range(J):
        for k in range(K):
            a = A(lam[k, i])
            weight_edge[k, i] = a.get_expectation_of_log()

    return np.exp(weight_edge)


# Returns an array of weights, K X J X M
# Normalization: dividing by average
# Working with logs
def update_weight_vertex(lam, pi, theta, tau, alpha_gam, beta_gam, cells):
    K = len(pi[0])
    J = len(lam[0, 0, :])
    N = len(cells[:, 0])
    M = len(cells[0, :])
    weight_vertex = np.zeros((K, J, M))

    for k in range(K):
            for j in range(J):
                weight_vertex[k, j, :] += np.sum([pi[n, k] *
                                                  calculate_expectation_of_D(j, theta[n], tau[n], alpha_gam, beta_gam, cells[n])
                                                  for n in range(N)], axis=0)

    # normalization
    for k in range(K):
        weight_vertex[k] = weight_vertex[k] - (np.sum(weight_vertex[k], axis=0) / J)

    return weight_vertex


def update_theta_tau(params):
    theta, tau, pi_prev, expected_hidden, alpha_prev, beta_prev, cell = params
    sum, sum_2 = 0, 0
    M = len(expected_hidden[0, 0, :])
    J = len(expected_hidden[0, :, 0])
    K = len(expected_hidden[:, 0, 0])
    for k in range(K):
        for j in range(J):
            for m in range(M):
                sum += pi_prev[k] * expected_hidden[k, j, m] * (j ** 2)
                sum_2 += pi_prev[k] * expected_hidden[k, j, m] * j * cell[m]
    numerator = (theta / tau) + (sum_2 * alpha_prev / beta_prev)
    denom = (1 / tau) + (sum * alpha_prev / beta_prev)
    new_theta = numerator / denom
    new_tau = 1 / denom
    return new_theta, new_tau


class Prior(object):
    def __init__(self, delta, theta, tau, alpha_gam, beta_gam, lam):
        self.delta = delta
        self.theta = theta
        self.tau = tau
        self.alpha_gam = alpha_gam
        self.beta_gam = beta_gam
        self.lam = lam


class Params(object):
    def __init__(self, trans, delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex):
        self.delta = delta
        self.theta = theta
        self.tau = tau
        self.alpha_gam = alpha_gam
        self.beta_gam = beta_gam
        self.lam = lam
        self.pi = pi
        self.weight_initial = weight_initial
        self.weight_edge = weight_edge
        self.weight_vertex = weight_vertex
        self.trans = trans
        self.M = len(self.weight_vertex[0, 0, :])
        self.N = len(self.pi[:, 0])
        self.K = len(self.pi[0, :])
        self.J = len(self.lam[0, 0, :])

    # Returns a tuple
    def print(self):
        return (self.trans, self.delta, self.theta, self.tau, self.alpha_gam, self.beta_gam, self.lam, self.pi,
                 self.weight_initial, self.weight_edge, self.weight_vertex)

    def get_term1_inner(self, params):
        pi, theta, tau, c, expected_hidden = params
        res = 0
        for m in tqdm.tqdm(range(self.M)):
            for k in range(self.K):
                for j in range(self.J):
                    res += pi[k] * expected_hidden[k, j, m] * \
                           calculate_expectation_of_D(j, theta, tau, self.alpha_gam, self.beta_gam, c[m])
        return res

    # E[logP(Y,Z,C,Ψ)] = E[ logP(Y|Z,C,Ψ) + logP(C|Ψ) + logP(Z|Ψ) + logP(Ψ) ]
    def calculate_expectation_full_joint_probability(self, cells, prior, clusters):
        result = 0

        expected_hidden = np.zeros((self.K, self.J, self.M))
        sum_of_expected_hidden_two = np.zeros((self.K, self.J, self.J))
        for k in range(self.K):
            sum_of_expected_hidden_two[k] = clusters[k].sum_of_expectation_two()
            expected_hidden[k] = clusters[k].get_expectation()

        # expectation of log[p(Y|Z,C,Ψ)]
        def get_term1():
            import multiprocessing as mp
            pool = mp.Pool(1) #mp.Pool(mp.cpu_count())
            res = pool.map(self.get_term1_inner, [(self.pi[n], self.theta[n], self.tau[n], cells[n], expected_hidden)
                                                  for n in range(self.N)])
            pool.close()
            return np.sum(res)

        # expectation of log[p(C|Ψ)]
        def get_term2():
            term = 0
            for k in range(self.K):
                for i in range(self.J):
                    a = A(self.lam[k, i])
                    term += np.sum(sum_of_expected_hidden_two[k, i, :] * a.get_expectation_of_log())
            return term

        # expectation of log[p(Z|Ψ)]
        def get_term3():
            res = 0
            pi = Pi(self.delta)
            for n in range(self.N):
                res += np.sum(self.pi[n, :] * pi.get_expectation_of_log())
            return res

        # expectation of log[P(A)]
        def get_term5():
            res = 0
            for k in range(self.K):
                for i in range(self.J):
                    a = A(self.lam[k, i])
                    term1 = np.sum((prior.lam[k, i, :] - 1) * a.get_expectation_of_log())
                    res += term1
            return res

        # expectation of log[P(μ)]
        def get_term6():
            res = 0
            for n in range(self.N):
                norm = Gaussian(self.theta[n], self.tau[n])
                E_mu_n = norm.get_expectation_mu()
                E_mu2_n = norm.get_expectation_mu_square()
                res += math.log(2 * math.pi * prior.tau) + (E_mu2_n - 2 * prior.theta * E_mu_n + prior.theta ** 2) / prior.tau
            return - .5 * res

        # expectation of log[P(π)]
        def get_term7():
            pi = Pi(self.delta)
            term1 = np.sum((prior.delta - 1) * pi.get_expectation_of_log())
            return term1

        # expectation of log[P(1/σ2)]
        def get_term8():
            gam = Gamma(self.alpha_gam, self.beta_gam)
            E_sigma_log_1sigma2 = gam.get_expectation_of_log()
            E_sigma_1sigma2 = gam.get_expectation()
            return (-prior.beta_gam) * E_sigma_1sigma2 + (prior.alpha_gam -1) * E_sigma_log_1sigma2

        # expectation of log[p(Y|Z,C,Ψ)]
        result += get_term1()

        # expectation of log[p(C|Ψ)]
        result += get_term2()

        # expectation of log[p(Z|Ψ)]
        result += get_term3()

        # expectation of log[p(Ψ)] which is log[P(A)] + log[P(θ)] + log[P(π)]
        ##########################

        # expectation of log[P(A)]
        result += get_term5()

        # expectation of log[P(μ)]
        result += get_term6()

        # expectation of log[P(π)]
        result += get_term7()

        # expectation of log[P(1/σ2)]
        result += get_term8()

        return result

    # E[log(q(Z,C,Ψ))] = E[ ∑n=1 log(q(Z_n)) + ∑k=1 log(q(C_k)) +
    #                       ∑k=1∑i=1 log(q(a_ki)) + ∑n=1 log(q(μ_n)) + log(q(π)) + log[q(1/σ2)]]
    def calculate_expectation_entropy(self, clusters):
        result = 0

        # expectation of ∑n=1 log(q(Z_n))
        def get_term1():
            res = 0
            for n in range(self.N):
                cluster = Z(self.pi[n])
                res += cluster.get_expectation_of_log_of_probability()
            return res

        # expectation of ∑k=1 log(q(C_k))
        def get_term2():
            res = 0
            for k in range(self.K):
                res += clusters[k].get_expectation_of_log_of_probability()
            return res

        # expectation of ∑k=1∑i=1 log(q(a_ki))
        def get_term4():
            res = 0
            for k in range(self.K):
                for i in range(self.J):
                    a = A(self.lam[k, i])
                    term1 = np.sum((self.lam[k, i, :] - 1) * a.get_expectation_of_log())
                    res += term1
            return res

        # expectation of ∑n=1 log(q(μ_n))
        def get_term5():
            res = 0
            for n in range(self.N):
                res += math.log(math.sqrt(math.pi * 2 * self.tau[n])) + 1
            return -.5 * res

        # expectation of log(q(π))
        def get_term6():
            pi = Pi(self.delta)
            return np.sum(pi.get_expectation_of_log() * (self.delta - 1))

        # expectation of log[q(1/σ2)
        def get_term7():
            gam = Gamma(self.alpha_gam, self.beta_gam)
            res = math.log(self.beta_gam) - .5 * (1 - 1/self.alpha_gam)
            return res

        # expectation of ∑n=1 log(q(Z_n))
        result += get_term1()

        # expectation of ∑k=1 log(q(C_k))
        result += get_term2()

        # expectation of ∑k=1∑i=1 log(q(a_ki))
        result += get_term4()

        # expectation of ∑n=1 log(q(μ_n))
        result += get_term5()

        # expectation of log(q(π))
        result += get_term6()

        # expectation of log[q(1/σ2)
        result += get_term7()

        return result

    def get_elbo(self, cells, prior, clusters):
        return self.calculate_expectation_full_joint_probability(cells, prior, clusters) - \
               self.calculate_expectation_entropy(clusters)


def calculate_most_probable_states(cells, trans, emiss, weight_initial, pi):
    M = len(trans[0, 0, :])
    J = len(trans[0, :, 0])
    K = len(trans[:, 0, 0])
    N = len(cells[:, 0])
    states = np.zeros((K, M))
    log_probs = np.zeros((K, M))

    for k in range(K):
        model = StubHMM(J)
        model.transmat_ = trans[k]
        model.startprob_ = weight_initial[k]
        model.framelogprob = np.swapaxes(emiss[k], 0, 1)
        log_probs[k], states[k] = model.decode(cells, lengths=np.ones(N)*M, algorithm="viterbi")

    for n in range(N):
        index = np.where(pi == np.max(pi))[0][0]

    return log_probs, states


# Requires numpy arrays (do np.array() before passing arguments)
def vi(locs, prior, init, y, max_iter=15, tol=.000000001):
    N = len(y[:, 0])
    import multiprocessing as mp

    delta_prior, theta_prio, tau_prior, alpha_gam_prior, beta_gam_prior, lam_prior = prior
    delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex = init
    prior = Prior(delta_prior, theta_prio, tau_prior, alpha_gam_prior, beta_gam_prior, lam_prior)

    print(pi)

    # Updating alpha_gam outside the iterations since it is only based on cells/reads/y
    # Update 1
    new_alpha_gam = update_alpha_gam(prior.alpha_gam, y)

    init_params = Params(weight_edge, delta, theta, tau, new_alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)

    p_list = [init_params]

    K = len(pi[0])
    J = len(lam[0, 0, :])
    M = len(y[0])
    m = 0
    trans = np.zeros((K, J, J))

    clusters = []
    for k in range(K):
        clusters.insert(k, C(M, weight_initial[k], weight_edge[k], weight_vertex[k]))

    while m < max_iter:
        clusters_prev = clusters.copy()
        expected_hidden = np.zeros((K, J, M))
        sum_of_expected_hidden_two = np.zeros((K, J, J))
        new_lam = np.zeros((K, J, J))
        p_prev = p_list[m]

        # Update 2
        new_delta = update_delta(prior.delta, p_prev.pi)

        for k in range(K):
            # handle chromosomal loci
            p_prev.weight_vertex[k] = correct_for_chrom(p_prev.weight_vertex[k], locs)
            graph = C(M, p_prev.weight_initial[k], p_prev.weight_edge[k], p_prev.weight_vertex[k])
            # Update 3
            clusters.insert(k, graph)
            sum_of_expected_hidden_two[k] = clusters[k].sum_of_expectation_two()

            trans[k] = graph.hmm.transmat_

            for i in range(J):
                # Update 4
                new_lam[k, i] = update_lambda(prior.lam[k, i, :], sum_of_expected_hidden_two[k, i])

            # Update 5
            expected_hidden[k] = clusters[k].get_expectation()

        # Update 6
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(update_theta_tau, [(prior.theta, prior.tau, p_prev.pi[n], expected_hidden, new_alpha_gam, p_prev.beta_gam, y[n]) for n in range(N)])
        result_asarray = np.array(result)
        new_theta, new_tau = result_asarray[:,0], result_asarray[:,1]
        pool.close()

        # Update 7
        new_beta_gam = update_beta_gam((prior.beta_gam, p_prev.pi, expected_hidden, new_theta, new_tau, y))

        # Update 8
        new_pi = update_pi(new_delta, expected_hidden, new_theta, new_tau, new_alpha_gam, new_beta_gam, y)

        # Update 9
        new_weight_vertex = update_weight_vertex(new_lam, new_pi, new_theta, new_tau, new_alpha_gam, new_beta_gam, y)
        new_weight_edge = update_weight_edge(new_lam, new_pi)

        p_next = Params(trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi,
                        weight_initial, new_weight_edge, new_weight_vertex)
        p_list.append(p_next)

        # ELBO
        elbo_values_prev = p_prev.get_elbo(y, prior, clusters_prev)
        elbo_values_next = p_next.get_elbo(y, prior, clusters)
        print(new_pi)
        print("ELBO "+str(elbo_values_prev))
        if (elbo_values_next - elbo_values_prev) <= tol:
            break

        print(m)
        m += 1

    l = p_list[-1]
    vi.likelihood = elbo_values_prev
    return l.print()



