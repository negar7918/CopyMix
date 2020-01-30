from scipy.stats import dirichlet, multinomial, gamma, poisson
import numpy as np
import math
from hmmlearn.base import _BaseHMM
from hmmlearn import utils
from scipy.special import digamma, logsumexp, factorial, gamma as gamma_func


def beta_func(x):
    return np.prod(gamma_func(x)) / gamma_func(np.sum(x))


def add_noise(j):
    if j == 0:
        j = 0.001
    return j


class Dirichlet:
    def __init__(self, alphas):
        self.alphas = alphas

    def get_distribution(self):
        return dirichlet(alpha=self.alphas)

    def get_expectation(self, index):
        return self.alphas[index] / np.sum(self.alphas)

    def get_expectation_of_log(self):
        return digamma(self.alphas) - digamma(np.sum(self.alphas))


class Categorical:
    def __init__(self, p):
        self.n = 1
        self.p = p

    def get_distribution(self):
        return multinomial(self.n, self.p)

    def get_expectation_of_log_of_probability(self):
        # E[log(∏k=1 p_k ^ I(Z_n=k))] = ∑k=1 E[I(Z_n=k)] * log(p_k) = ∑k=1 p_k * log(p_k)
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


class Rho(Dirichlet):
    def __init__(self, *args, **kwargs):
        Dirichlet.__init__(self, *args, **kwargs)


class A(Dirichlet):
    def __init__(self, *args, **kwargs):
        Dirichlet.__init__(self, *args, **kwargs)


class Theta:
    # epsilon_s = alpha
    # epsilon_r = beta
    # scale = 1 / beta
    def __init__(self, epsilon_r, epsilon_s):
        self.a = epsilon_s
        self.scale = 1 / epsilon_r

    def get_distribution(self):
        return gamma(a=self.a, scale=self.scale)

    def get_expectation(self):
        return self.a * self.scale

    def get_expectation_of_log(self):
        return digamma(self.a) - math.log(1 / self.scale) # + .0000001)


class StubHMM(_BaseHMM):
    # An HMM with hardcoded observation probabilities
    def _compute_log_likelihood(self, X):
        return self.framelogprob


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
        h.framelogprob = self.observation_prob
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

    # Returns an array of probabilities, J X J
    # ∑m=2 E(I(C_km=j, C_km-1))
    def sum_of_expectation_two(self):
        state_probability = np.zeros((self.J, self.J, self.M-1))
        # HMM: fwd * emission * transition * bwd / prob(observations)
        # Directed Graph: fwd_{m-1} * transition * emission_{m} * bwd_{m}

        for i in range(self.J):
            for j in range(self.J):
                for m in range(1, self.M):
                    state_probability[i, j, m-1] = self.fwd[m-1, j] + np.log(self.hmm.transmat_[i, j]) +\
                                              self.observation_prob[m, j] + self.bwd[m, j]

        result = np.exp(state_probability - logsumexp(state_probability, axis=(0, 1)))  # log normalization

        return np.sum(result, axis=2)

    def get_expectation_of_log_of_probability(self):
        # E[ ∑m=1 ∑j=1 I(C_km=j)v_weight + ∑j=1 I(Ck1=j) init_weight + ∑m=2 ∑j=1 ∑i=1 I(Ckm−1=i,Ckm=j) edge_weight
        expected_hidden = self.get_expectation()
        sum_of_expected_hidden_two = self.sum_of_expectation_two()
        term1 = np.sum(expected_hidden * np.swapaxes(self.observation_prob, 0, 1))
        term2 = np.sum(expected_hidden[:, 0] * self.hmm.startprob_)
        term3 = np.sum(sum_of_expected_hidden_two * self.hmm.transmat_)
        return term1 + term2 + term3


# Returns an array of updated alpha
def update_alpha(alpha, pi_prev):
    sum_of_expectations = [np.sum(pi_prev[:, k]) for k in range(len(pi_prev[0]))]
    return alpha + sum_of_expectations


# Returns an array of updated delta
def update_delta(delta, expectation):
    return delta + expectation


# Returns an array of updated lambda
def update_lambda(lam, sum_of_expectation_two):
    return lam + sum_of_expectation_two


def update_epsilon_r_with_rates(params):
    epsilon_s, rate = params
    return epsilon_s / rate


# Returns the updated epsilon_r
def update_epsilon_r(params):
    epsilon, pi_prev, expected_hidden = params
    sums = 0
    J = len(expected_hidden[0, :, 0])
    M = len(expected_hidden[0, 0, :])
    for m in range(M):
        for j in range(J):
            sums += np.sum(pi_prev[:] * expected_hidden[:, j, m] * add_noise(j))
    return epsilon + sums


# Returns the updated epsilon_s
def update_epsilon_s(epsilon_s, cells):
    return epsilon_s + np.sum(cells, axis=1)


# Returns an array of updated pi
def update_pi(params):
    alpha, expected_hidden, epsilon_r, epsilon_s, cell= params
    K = alpha.size
    J = len(expected_hidden[0, :, 0])
    sums = np.zeros(K)
    pi = Pi(alpha)

    for j in range(J):
        sums += np.sum((expected_hidden[:, j, :] * calculate_expectation_of_D(j, epsilon_r, epsilon_s, cell)), axis=1)

    updated_pi = pi.get_expectation_of_log() + sums

    return np.exp(updated_pi - logsumexp(updated_pi)) # log normalization


# Handles inf values by replacing them to the max value
def logfactorial(array):
    result = np.log(factorial(array))
    import sys
    replace = np.isinf(result)
    result[replace] = sys.maxsize
    return result


def replace_inf(array):
    import sys
    replace = np.isinf(array)
    if (array[replace] < 0).all():
        array[replace] = - sys.maxsize
    elif (array[replace] > 0).all():
        array[replace] = sys.maxsize
    return array


# Returns an array of M: expected value of D at j, n for all m in M
def calculate_expectation_of_D(j, epsilon_r, epsilon_s, cell):
    state = add_noise(j)

    if epsilon_r is None:
        return cell[:] * math.log(state) - state - logfactorial(cell[:])
    else:
        theta = Theta(epsilon_r, epsilon_s)
        return replace_inf(np.log(poisson.pmf(cell[:], theta.get_expectation() * state) + .0000001)
                                  + cell[:] * (digamma(epsilon_s) - math.log(epsilon_s)))


# Returns an array of weights, K X J
# No normalization needed since these are weights and not probabilities
def update_weight_initial(delta, pi):
    K = len(pi[0])
    J = len(delta[0])
    initial = np.zeros((K, J))
    for k in range(K):
        rho = Rho(delta[k])
        initial[k] = rho.get_expectation_of_log()

    return np.exp(initial)


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
def update_weight_vertex(lam, pi, epsilon_r, epsilon_s, cells):
    K = len(pi[0])
    J = len(lam[0, 0, :])
    N = len(cells[:, 0])
    M = len(cells[0, :])
    weight_vertex = np.zeros((K, J, M))

    if epsilon_r is None:
        for k in range(K):
                for j in range(J):
                    weight_vertex[k, j, :] += np.sum([pi[n, k] *
                                                      calculate_expectation_of_D(j, None, None, cells[n])
                                                      for n in range(N)], axis=0)

    else:
        for k in range(K):
            for j in range(J):
                weight_vertex[k, j, :] += np.sum([pi[n, k] *
                                                  calculate_expectation_of_D(j, epsilon_r[n], epsilon_s[n], cells[n])
                                                  for n in range(N)], axis=0)

    # normalization
    for k in range(K):
        weight_vertex[k] = weight_vertex[k] - (np.sum(weight_vertex[k], axis=0) / J)

    return weight_vertex


class Params(object):
    def __init__(self, alpha, epsilon_r, delta, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, posterior):
        self.alpha = alpha
        self.epsilon_r = epsilon_r
        self.delta = delta
        self.lam = lam
        self.pi = pi
        self.epsilon_s = epsilon_s
        self.weight_initial = weight_initial
        self.weight_edge = weight_edge
        self.weight_vertex = weight_vertex
        self.posterior = posterior
        self.M = len(self.weight_vertex[0, 0, :])
        self.N = len(self.pi[:, 0])
        self.K = len(self.pi[0, :])
        self.J = len(self.lam[0, 0, :])

    # Returns a tuple
    def print(self):
        return (self.alpha, self.epsilon_r, self.delta, self.lam, self.pi, self.epsilon_s,
                 self.weight_initial, self.weight_edge, self.weight_vertex, self.posterior)

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
            res = 0
            if self.epsilon_r is not None:
                for n in range(self.N):
                    for k in range(self.K):
                        for j in range(self.J):
                            res += np.sum(self.pi[n, k] * expected_hidden[k, j, :] *
                                   calculate_expectation_of_D(j, self.epsilon_r[n], self.epsilon_s[n], cells[n]))
            else:
                for n in range(self.N):
                    for k in range(self.K):
                        for j in range(self.J):
                            res += np.sum(self.pi[n, k] * expected_hidden[k, j, :] *
                                   calculate_expectation_of_D(j, None, None, cells[n]))
            return res

        # expectation of log[p(C|Ψ)]
        def get_term2():
            term1, term2 = 0, 0
            for k in range(self.K):
                if self.delta is not None:
                    rho = Rho(self.delta[k])
                    term1 += np.sum(expected_hidden[k, :, 0] * rho.get_expectation_of_log())
                for i in range(self.J):
                    a = A(self.lam[k, i])
                    term2 += np.sum(sum_of_expected_hidden_two[k, i, :] * a.get_expectation_of_log())
            return term1 + term2

        # expectation of log[p(Z|Ψ)]
        def get_term3():
            res = 0
            pi = Pi(self.alpha)
            for n in range(self.N):
                res += np.sum(self.pi[n, :] * pi.get_expectation_of_log())
            return res

        # expectation of log[P(ρ)]
        def get_term4():
            res = 0
            if self.delta is not None:
                for k in range(self.K):
                    rho = Rho(self.delta[k])
                    term1 = np.sum(prior.delta[k, :] * rho.get_expectation_of_log())
                    res += term1 - math.log(beta_func(prior.delta[k]))
            return res

        # expectation of log[P(A)]
        def get_term5():
            res = 0
            for k in range(self.K):
                for i in range(self.J):
                    a = A(self.lam[k, i])
                    term1 = np.sum((prior.lam[k, i, :] - 1) * a.get_expectation_of_log())
                    res += term1 + math.log(beta_func(prior.lam[k, i]) + .00000000001)
            return res

        # expectation of log[P(θ)]
        def get_term6():
            res = 0
            if self.epsilon_r is not None:
                for n in range(self.N):
                    theta = Theta(self.epsilon_r[n], self.epsilon_s[n])
                    res += (prior.epsilon_s[n] - 1) * theta.get_expectation_of_log() - prior.epsilon_r[
                                                                                          n] * theta.get_expectation()
            return res

        # expectation of log[P(π)]
        def get_term7():
            pi = Pi(self.alpha)
            term1 = np.sum((prior.alpha - 1) * pi.get_expectation_of_log())
            return term1 - math.log(beta_func(prior.alpha) + .00000000001)

        # expectation of log[p(Y|Z,C,Ψ)]
        result += get_term1()

        # expectation of log[p(C|Ψ)]
        result += get_term2()

        # expectation of log[p(Z|Ψ)]
        result += get_term3()

        # expectation of log[p(Ψ)] which is log[P(ρ)] + log[P(A)] + log[P(θ)] + log[P(π)]
        ##########################
        # expectation of log[P(ρ)]
        result += get_term4()

        # expectation of log[P(A)]
        result += get_term5()

        # expectation of log[P(θ)]
        result += get_term6()

        # expectation of log[P(π)]
        result += get_term7()

        return result

    # E[log(q(Z,C,Ψ))] = E[ ∑n=1 log(q(Z_n)) + ∑k=1 log(q(C_k)) + ∑k=1 log(q(ρ_k)) +
    #                       ∑k=1∑i=1 log(q(a_ki)) + ∑n=1 log(q(θ_n)) + log(q(π)) ]
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

        # expectation of ∑k=1 log(q(ρ_k))
        def get_term3():
            res = 0
            if self.delta is not None:
                for k in range(self.K):
                # E[ ∑j=1 E(I(C_k1=j)) log(ρ_kj) + ∑j=1 (δj−1) log(ρ_kj)] =
                # ∑j=1 E[log(ρ_kj)] (δ∗−1)
                    rho = Rho(self.delta[k])
                    res += np.sum(rho.get_expectation_of_log() * (self.delta[k, :] - 1))
            return res

        # expectation of ∑k=1∑i=1 log(q(a_ki))
        def get_term4():
            res = 0
            for i in range(self.J):
                for k in range(self.K):
                    a = A(self.lam[k, i])
                    res += np.sum(a.get_expectation_of_log() * (self.lam[k, i, :] - 1))
            return res

        # expectation of ∑n=1 log(q(θ_n))
        def get_term5():
            res = 0
            if self.epsilon_r is not None:
                for n in range(self.N):
                    t = Theta(self.epsilon_r[n], self.epsilon_s[n])
                    res += t.get_expectation_of_log() * self.epsilon_s[n] - t.get_expectation() * self.epsilon_r[n] - 1
            return res

        # expectation of log(q(π))
        def get_term6():
            pi = Pi(self.alpha)
            return np.sum(pi.get_expectation_of_log() * (self.alpha - 1))

        # expectation of ∑n=1 log(q(Z_n))
        result += get_term1()

        # expectation of ∑k=1 log(q(C_k))
        result += get_term2()

        # expectation of ∑k=1 log(q(ρ_k))
        result += get_term3()

        # expectation of ∑k=1∑i=1 log(q(a_ki))
        result += get_term4()

        # expectation of ∑n=1 log(q(θ_n))
        result += get_term5()

        # expectation of log(q(π))
        result += get_term6()

        return result

    def get_elbo(self, cells, prior, clusters):
        return self.calculate_expectation_full_joint_probability(cells, prior, clusters) - \
               self.calculate_expectation_entropy(clusters)

    def get_dic(self, clusters, cells):
        # calculate expected_hidden
        expected_hidden = np.zeros((self.K, self.J, self.M))
        sum_of_expected_hidden_two = np.zeros((self.K, self.J, self.J))
        for k in range(self.K):
            sum_of_expected_hidden_two[k] = clusters[k].sum_of_expectation_two()
            expected_hidden[k] = clusters[k].get_expectation()

        def handle_inf(x):
            import sys
            if np.isinf(x):
                return sys.maxsize
            else:
                return x

        # term_1 : -4 E[ log[p(Y | Z, C, Ψ)] ] w.r.t. final posterior values
        # term_2 : 2 log[p(Y|Z,C,Ψ)] where Z, C and Ψ are the modes (maximizing the posteriors)
        res = 0
        for n in range(self.N):
            for k in range(self.K):
                for j in range(self.J):
                    res += np.sum(self.pi[n, k] * expected_hidden[k, j, :] *
                                  calculate_expectation_of_D(j, self.epsilon_r[n], self.epsilon_s[n], cells[n]))
        term_1 = - res

        res = 0
        for n in range(self.N):
            for m in range(self.M):
                k = int(np.argmax(self.pi[n, :]))
                j = int(np.argmax(expected_hidden[k, :, m]))
                theta = self.epsilon_s[n] / self.epsilon_r[n]
                state = add_noise(j)
                D = handle_inf(math.log(poisson.pmf(cells[n, m], theta * state) + .0000001))
                res += D
        term_2 = 2 * res

        return term_1, 4 * term_1 + term_2


# Requires numpy arrays (do np.array() before passing arguments)
def vi(params, y, max_iter=100, tol=.001):
    N = len(y[:, 0])
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())

    alpha, epsilon_r, delta, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, rates = params
    prior = Params(alpha, epsilon_r, delta, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, weight_vertex)

    # Updating epsilon_s outside the iterations since it is only based on cells/reads/y
    e_s = np.zeros(len(y))
    if epsilon_s is not None:
        e_s = update_epsilon_s(epsilon_s, y)
        if rates is not None:
            epsilon_r = pool.map(update_epsilon_r_with_rates,
                                 [(e_s[n], rates[n])  # (epsilon_r[n], p_prev.pi[n], expected_hidden)##
                                  for n in range(N)])

    if delta is not None:
        weight_initial = update_weight_initial(delta, pi)
    weight_edge = update_weight_edge(lam, pi)
    weight_vertex = update_weight_vertex(lam, pi, epsilon_r, e_s, y)

    init_params = Params(alpha, epsilon_r, delta, lam, pi, e_s, weight_initial, weight_edge, weight_vertex, weight_vertex)
    p_list = [init_params]

    K = len(pi[0])
    J = len(lam[0, 0, :])
    M = len(y[0])
    m = 0

    clusters = []
    for k in range(K):
        clusters.insert(k, C(M, weight_initial[k], weight_edge[k], weight_vertex[k]))

    while m < max_iter:
        clusters_prev = clusters.copy()
        expected_hidden = np.zeros((K, J, M))
        sum_of_expected_hidden_two = np.zeros((K, J, J))
        new_lam = np.zeros((K, J, J))
        new_delta = np.zeros((K, J))
        new_epsilon_r = np.zeros(len(y))
        p_prev = p_list[m]

        # Update 1
        new_alpha = update_alpha(alpha, p_prev.pi)

        for k in range(K):
            # Update 2
            clusters.insert(k, C(M, p_prev.weight_initial[k], p_prev.weight_edge[k], p_prev.weight_vertex[k]))
            sum_of_expected_hidden_two[k] = clusters[k].sum_of_expectation_two()

            for i in range(J):
                # Update 4
                new_lam[k, i] = update_lambda(lam[k, i, :], sum_of_expected_hidden_two[k, i])

            # Update 2
            expected_hidden[k] = clusters[k].get_expectation()
            # Update 3
            if delta is not None:
                new_delta[k] = update_delta(delta[k, :], expected_hidden[k, :, 0])

        if epsilon_r is not None:
            if rates is not None:
                # Update 5
                new_epsilon_r = pool.map(update_epsilon_r_with_rates, [(e_s[n], rates[n]) for n in range(N)])
            else:
                # Update 5
                new_epsilon_r = pool.map(update_epsilon_r, [(epsilon_r[n], p_prev.pi[n], expected_hidden) for n in
                                                            range(N)])
            # Update 6
            new_pi = np.asarray(pool.map(update_pi, [(new_alpha, expected_hidden, new_epsilon_r[n], e_s[n], y[n])
                                                     for n in range(N)]))
            # Update 7
            new_weight_vertex = update_weight_vertex(new_lam, new_pi, new_epsilon_r, e_s, y)
        else:
            new_pi = np.asarray(pool.map(update_pi, [(new_alpha, expected_hidden, None, None, y[n]) for n in range(N)]))
            new_weight_vertex = update_weight_vertex(new_lam, new_pi, None, None, y)

        # Update 7
        new_weight_initial = weight_initial
        if delta is not None:
            new_weight_initial = update_weight_initial(delta, pi)
        new_weight_edge = update_weight_edge(new_lam, new_pi)

        p_next = Params(new_alpha, (None if epsilon_r is None else new_epsilon_r),
                        (None if delta is None else new_delta), new_lam, new_pi,
                        (None if epsilon_s is None else e_s), new_weight_initial,
                        new_weight_edge, new_weight_vertex, expected_hidden)
        p_list.append(p_next)

        # ELBO
        elbo_values_prev = p_prev.get_elbo(y, prior, clusters_prev)
        elbo_values_next = p_next.get_elbo(y, prior, clusters)
        print(np.abs(elbo_values_prev - elbo_values_next))
        print(new_pi)
        if np.abs(elbo_values_prev - elbo_values_next) <= tol:
            break

        print(m)
        m += 1

    pool.close()

    l = p_list[-1]
    vi.likelihood, vi.dic = l.get_dic(clusters, y)
    return l.print()



