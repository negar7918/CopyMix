from CopyMix.CopyMix_Gaussian.inference import vi
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import dirichlet
import random
import tqdm


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
        pi[n] = generate_categorical_prob(num_of_clusters) # generate_categorical_prob(num_of_clusters, 10)
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
            weight_edge[c, s] = generate_categorical_prob(num_of_states, .1)
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


class Initialization(object):
    def __init__(self):
        self.delta = None
        self.theta = None
        self.tau = None
        self.alpha_gam = None
        self.beta_gam = None
        self.lam = None
        self.pi = None
        self.weight_initial = None
        self.weight_edge = None
        self.weight_vertex = None
        self.method = None
        self.likelihood = None
        self.num_of_clusters = None

    def add_init(self, delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex):
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

    def add_new(self, trans, delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex, likelihood,
                method, num_of_clusters):
        self.trans = trans
        self.new_delta = delta
        self.new_theta = theta
        self.new_tau = tau
        self.new_alpha_gam = alpha_gam
        self.new_beta_gam = beta_gam
        self.new_lam = lam
        self.new_pi = pi
        self.new_weight_initial= weight_initial
        self.new_weight_edge = weight_edge
        self.new_weight_vertex = weight_vertex
        self.method = method
        self.likelihood = likelihood
        self.num_of_clusters = num_of_clusters

    def save(self):
        return np.asarray([self.trans,
        self.delta,
        self.theta,
        self.tau,
        self.alpha_gam,
        self.beta_gam,
        self.lam,
        self.pi,
        self.weight_initial,
        self.weight_edge,
        self.weight_vertex,
        self.new_delta,
        self.new_theta,
        self.new_tau,
        self.new_alpha_gam,
        self.new_beta_gam,
        self.new_lam,
        self.new_pi,
        self.new_weight_edge,
        self.new_weight_vertex,
        self.method,
        self.likelihood,
        self.num_of_clusters])


def generate_initializations(id, data, num_of_states, total_num_of_clusters):
    inits = [Initialization() for i in range(50)]
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
    theta_prior = 0
    tau_prior = 1
    alpha_prior = 0
    beta_prior = 0

    theta = np.ones(num_of_cells)
    tau = np.ones(num_of_cells)
    for n in range(num_of_cells):
        theta[n] = np.mean(data[n])  # mean of data # 10
        tau[n] = np.var(data[n])  # var of data # 1
    alpha_gam = 1
    beta_gam = np.var(data)

    index = 0

    for k in range(total_num_of_clusters, total_num_of_clusters+1): #range(1, total_num_of_clusters+1):

        delta = 10 * np.ones(k)

        for i in tqdm.tqdm(range(25)):

            pi, classes = get_clustering_random(k, data)
            weight_initial, weight_edge, weight_vertex = get_weights_random(i, k, num_of_states, data)
            lam = weight_edge * 100 + .0000000000001
            prior = (delta, theta_prior, tau_prior, alpha_prior, beta_prior, lam)
            init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
            inits[index].add_init(delta, theta, tau, alpha_gam, beta_gam, lam, pi,
                                  weight_initial, weight_edge, weight_vertex)
            new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
            new_weight_vertex, trans = vi(prior, init, data)
            inits[index].add_new(trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
            new_weight_vertex, vi.likelihood, "random_random"+str(i), k)

            index += 1

            pi, weight_initial, weight_edge, weight_vertex = get_clustering_transitions_random(i, k, num_of_states, data)
            lam = weight_edge * 100 + .0000000000001
            prior = (delta, theta_prior, tau_prior, alpha_prior, beta_prior, lam)
            init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
            inits[index].add_init(delta, theta, tau, alpha_gam, beta_gam, lam, pi,
                                 weight_initial, weight_edge, weight_vertex)
            new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
            new_weight_vertex, trans = vi(prior, init, data)
            inits[index].add_new(trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi,
                                 weight_initial, new_weight_edge, new_weight_vertex, vi.likelihood, "random" + str(i), k)

            print('inits ' + str(i) + ' are done.')

            index += 1

    list_of_inits = []
    for i in range(50):
        d = inits[i].save()
        list_of_inits = np.append(list_of_inits, d)

    import pickle
    f = open('./inits_' + id, 'wb')
    pickle.dump(list_of_inits, f)
    f.close()