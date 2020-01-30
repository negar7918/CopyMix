from CopyMix.inference import vi
from CopyMix import bw_poisson_one_lambda, bw_poisson, em_independent_copy_number
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import dirichlet
import random


def get_clustering_kmeans(num_of_clusters, data):
    num_of_cells = len(data[:, 0])
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0).fit(data)
    classes = kmeans.labels_
    pi = np.zeros((num_of_cells, num_of_clusters))
    for n in range(num_of_cells):
        pi[n, classes[n]] = 1
    return pi, classes


def get_clustering_random(num_of_clusters, data):
    num_of_cells = len(data[:, 0])
    pi = np.zeros((num_of_cells, num_of_clusters))
    classes = np.zeros(num_of_cells)
    for n in range(num_of_cells):
        pi[n] = generate_categorical_prob(num_of_clusters)
        classes[n] = np.where(pi[n] == max(pi[n]))[0][0]
    return pi, classes


def get_weights_random(i, num_of_clusters, num_of_states, data):
    np.random.seed(i)
    length = len(data[0, :])
    weight_initial = np.ones((num_of_clusters, num_of_states)) / num_of_clusters
    weight_edge = np.zeros((num_of_clusters, num_of_states, num_of_states))
    weight_vertex = np.ones((num_of_clusters, num_of_states, length)) / 100
    for c in range(num_of_clusters):
        weight_initial[c] = generate_categorical_prob(num_of_states)
        for s in range(num_of_states):
            weight_edge[c, s] = generate_categorical_prob(num_of_states)
            for l in range(length):
                weight_vertex[c, s, l] = random.uniform(0.1, 0.9)
    return weight_initial, weight_edge, weight_vertex


def generate_categorical_prob(num_of_categories, alpha=None):
    if alpha is None:
        alpha = np.random.mtrand.dirichlet([0.1] * num_of_categories)
    else:
        alpha = alpha * np.ones(num_of_categories)
    var = dirichlet.rvs(alpha=alpha, size=1, random_state=None)
    return var[0]


def get_clustering_transitions_random(i, num_of_clusters, num_of_states, data):
    np.random.seed(i)
    num_of_cells = len(data[:, 0])
    length = len(data[0, :])
    pi = np.zeros((num_of_cells, num_of_clusters))
    weight_initial = np.zeros((num_of_clusters, num_of_states))
    weight_edge = np.zeros((num_of_clusters, num_of_states, num_of_states))
    weight_vertex = np.zeros((num_of_clusters, num_of_states, length))

    for n in range(num_of_cells):
        pi[n] = generate_categorical_prob(num_of_clusters, alpha=10)

    counter = 0
    for c in range(num_of_clusters):
        weight_initial[c] = generate_categorical_prob(num_of_states, alpha=10)
        for s in range(num_of_states):
            weight_edge[c, s] = generate_categorical_prob(num_of_states, alpha=10)
            for l in range(length):
                random.seed(counter)
                weight_vertex[c, s, l] = random.uniform(0.1, 0.9)
                counter += 1

    return pi, weight_initial, weight_edge, weight_vertex


def get_transitions_bw(pi, classes, num_of_states, data, normal):
    num_of_clusters = len(pi[0, :])
    length = len(data[0, :])
    weight_initial = np.ones((num_of_clusters, num_of_states)) / num_of_states
    weight_edge = np.zeros((num_of_clusters, num_of_states, num_of_states))
    weight_vertex = np.ones((num_of_clusters, num_of_states, length)) / 100
    weight_vertex_new = np.zeros((num_of_clusters, num_of_states, length))

    np.random.seed(1)
    prng = np.random.RandomState(2)

    rates = [i * 4 for i in range(1, num_of_states+1)]
    for cluster in range(num_of_clusters):
        weight_edge[cluster] = prng.rand(num_of_states, num_of_states)
        normalized_edge = weight_edge[cluster] / np.sum(weight_edge[cluster], axis=1)[:, np.newaxis]
        if normal == True:
            hmm = bw_poisson.C(weight_initial[cluster], normalized_edge, weight_vertex[cluster],
                                          rates, length)
        else:
            hmm = bw_poisson_one_lambda.C(weight_initial[cluster], normalized_edge, weight_vertex[cluster],
                                          rates, length)
        hmm.get_EM_estimation(data[np.where(classes == cluster)])
        weight_initial[cluster], weight_edge[cluster], weight_vertex_new[cluster] = hmm.get_hmm()

    return weight_initial, weight_edge, weight_vertex_new


class Initialization(object):
    def __init__(self):
        self.alpha = None
        self.epsilon_r = None
        self.lam = None
        self.pi = None
        self.epsilon_s = None
        self.weight_initial = None
        self.weight_edge = None
        self.weight_vertex = None
        self.dic = None
        self.alpha_new = None
        self.epsilon_r_new = None
        self.lam_new = None
        self.pi_new = None
        self.epsilon_s_new = None
        self.weight_initial_new = None
        self.weight_edge_new = None
        self.weight_vertex_new = None
        self.method = None
        self.likelihood = None
        self.num_of_clusters = None
        self.post = None

    def add_init(self, alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex):
        self.alpha = alpha
        self.epsilon_r = epsilon_r
        self.lam = lam
        self.pi = pi
        self.epsilon_s = epsilon_s
        self.weight_initial = weight_initial
        self.weight_edge = weight_edge
        self.weight_vertex = weight_vertex

    def add_new(self, alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, post, dic, likelihood,
                method, num_of_clusters):
        self.alpha_new = alpha
        self.epsilon_r_new = epsilon_r
        self.lam_new = lam
        self.pi_new = pi
        self.epsilon_s_new = epsilon_s
        self.weight_initial_new = weight_initial
        self.weight_edge_new = weight_edge
        self.weight_vertex_new = weight_vertex
        self.post = post
        self.dic = dic
        self.method = method
        self.likelihood = likelihood
        self.num_of_clusters = num_of_clusters

    def save(self):
        return np.asarray([self.alpha, self.epsilon_r, self.lam, self.pi, self.epsilon_s, self.weight_initial,
                         self.weight_vertex, self.weight_edge, self.weight_vertex_new, self.pi_new, self.alpha_new,
                         self.dic, self.method, self.likelihood, self.num_of_clusters, self.post])


def generate_initializations(id, data, num_of_states, total_num_of_clusters):
    inits = [Initialization() for i in range(164)]
    num_of_cells = len(data)
    epsilon_r, epsilon_s = np.zeros(num_of_cells), np.zeros(num_of_cells)
    epsilon_r_informative, epsilon_s_informative = np.zeros(num_of_cells), np.zeros(num_of_cells)
    for n in range(num_of_cells):
        epsilon_s[n] = .001
        epsilon_r[n] = .001

    for n in range(num_of_cells):
        non_zeros = data[n, np.nonzero(data[n])]
        mean = np.mean(non_zeros)  # mean of data
        var = np.var(non_zeros)  # var of data
        epsilon_s_informative[n] = (mean * mean + .00001) / (var + .00001)
        epsilon_r_informative[n] = (mean + .00001)/ (var + .00001)

    index = 0

    for k in range(1, total_num_of_clusters+1):

        alpha = 10 * np.ones(k)

        pi, classes = get_clustering_kmeans(k, data)
        weight_initial, weight_edge, weight_vertex = get_transitions_bw(pi, classes, num_of_states, data, False)
        lam = weight_edge * 100 + .0000000000001
        # alpha, epsilon_r, delta= None, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, rate=None
        p_kmeans_bw_rate_times_state = alpha, epsilon_r, None, lam, pi, epsilon_s, weight_initial, weight_edge,\
                                       weight_vertex, None
        inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex)
        alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new,\
        weight_vertex_new, post = vi(p_kmeans_bw_rate_times_state, data)
        inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                             weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                             "kmeans_bw_rate_times_state", k)

        index += 1

        pi, classes = get_clustering_kmeans(k, data)
        weight_initial, weight_edge, weight_vertex = get_transitions_bw(pi, classes, num_of_states, data, True)
        lam = weight_edge * 100 + .0000000000001
        p_kmeans_bw = alpha, epsilon_r, None, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, None
        inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex)
        alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
        weight_vertex_new, post = vi(p_kmeans_bw, data)
        inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                             weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                             "kmeans_bw", k)

        index += 1

        pi, classes = get_clustering_random(k, data)
        weight_initial, weight_edge, weight_vertex = get_transitions_bw(pi, classes, num_of_states, data, True)
        lam = weight_edge * 100 + .0000000000001
        p_random_bw = alpha, epsilon_r, None, lam, pi, epsilon_s, weight_initial, weight_edge,\
                      weight_vertex, None
        inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge,
                              weight_vertex)
        alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
        weight_vertex_new, post = vi(p_random_bw, data)
        inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                             weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                             "random_bw", k)

        index += 1

        pi, classes = get_clustering_random(k, data)
        weight_initial, weight_edge, weight_vertex = get_transitions_bw(pi, classes, num_of_states, data, False)
        lam = weight_edge * 100 + .0000000000001
        p_random_bw_rate_times_state = alpha, epsilon_r, None, lam, pi, epsilon_s, \
                                       weight_initial, weight_edge, weight_vertex, None
        inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge,
                              weight_vertex)
        alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
        weight_vertex_new, post = vi(p_random_bw_rate_times_state, data)
        inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                             weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                             "random_bw_rate_times_state", k)

        index += 1

        a = np.zeros((k, num_of_states))
        for c in range(k):
            a[c] = generate_categorical_prob(num_of_states)
        theta = epsilon_s_informative / epsilon_r_informative
        pi_init = np.ones((num_of_cells, k)) / k
        pi, weight_vertex, weight_edge = em_independent_copy_number.EM(pi_init, a, theta, data)
        weight_initial = np.ones((k, num_of_states)) / num_of_states
        weight_edge = np.ones((k, num_of_states, num_of_states)) / num_of_states
        lam = weight_edge * 100 + .0000000000001
        p_independent_em = alpha, epsilon_r, None, lam, pi, epsilon_s, \
                                       weight_initial, weight_edge, weight_vertex, None
        inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge,
                              weight_vertex)
        alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
        weight_vertex_new, post = vi(p_independent_em, data)
        inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                             weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                             "independent_em", k)

        index += 1

        for i in range(12):

            pi, classes = get_clustering_random(k, data)
            weight_initial, weight_edge, weight_vertex = get_weights_random(i, k, num_of_states, data)
            lam = weight_edge * 100 + .0000000000001
            p_random_random = alpha, epsilon_r_informative, None, lam, pi, epsilon_s_informative, \
                                           weight_initial, weight_edge, weight_vertex, None
            inits[index].add_init(alpha, epsilon_r_informative, lam, pi, epsilon_s_informative, weight_initial, weight_edge,
                                  weight_vertex)
            alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
            weight_vertex_new, post = vi(p_random_random, data)
            inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                                 weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                                 "random_random", k)

            index += 1

            pi, classes = get_clustering_kmeans(k, data)
            weight_initial, weight_edge, weight_vertex = get_weights_random(i, k, num_of_states, data)
            lam = weight_edge * 100 + .0000000000001
            p_kmeans_random = alpha, epsilon_r, None, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, None
            inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex)
            alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
            weight_vertex_new, post = vi(p_kmeans_random, data)
            inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                                 weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                                 "kmeans_random", k)

            index += 1

            pi, weight_initial, weight_edge, weight_vertex = get_clustering_transitions_random(i, k, num_of_states, data)
            lam = weight_edge * 100 + .0000000000001
            p_random = alpha, epsilon_r, None, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex, None
            inits[index].add_init(alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_edge, weight_vertex)
            alpha_new, epsilon_r_new, delta, lam_new, pi_new, epsilon_s_new, weight_initial_new, weight_edge_new, \
            weight_vertex_new, post = vi(p_random, data)
            inits[index].add_new(alpha_new, epsilon_r_new, lam_new, pi_new, epsilon_s_new, weight_initial_new,
                                 weight_edge_new, weight_vertex_new, post, vi.dic, vi.likelihood,
                                 "random", k)

            index += 1

    list_of_inits = []
    for i in range(164):
        d = inits[i].save()
        list_of_inits = np.append(list_of_inits, d)

    import pickle
    f = open('./inits_' + id, 'wb')
    pickle.dump(list_of_inits, f)
    f.close()