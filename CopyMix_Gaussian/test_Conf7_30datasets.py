import numpy as np
import CopyMix.CopyMix_Gaussian.util as util
from CopyMix.CopyMix_Gaussian.inference import vi
from sklearn.metrics.cluster import v_measure_score
from scipy.stats import dirichlet
import math


def calculate_predicted_c(pi, weight_vertex, name):
    K = pi.shape[1]
    M = len(weight_vertex[0, 0, :])
    predicted_c = np.zeros((K, M))
    for i in range(K):
        predicted_c[i] = [np.argmax(weight_vertex[i, :, m]) for m in range(M)]

    return predicted_c

def run(s):
    rng = np.random.default_rng(s)
    rng = np.random.default_rng(s)
    num_of_cells = 150
    seq_len = 200
    trans_1 = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    trans_2 = np.array([[.8, .1, 0, 0, 0, .1], [0, .8, .1, 0, 0, .1], [0, 0, .9, .1, 0, 0], [0, 0, 0, .9, 0, .1], [0, 0, 0, 0, .9, .1], [0, 0, 0, 0, .1, .9]])
    trans_3 = np.array([[.9, .1, 0, 0, 0, 0], [0, .9, .1, 0, 0, 0], [.9, .1, 0, 0, 0, 0], [0, .9, 0, 0, 0, .1], [.9, 0, 0, 0, 0, .1], [0, 0, .1, 0, 0, .9]])
    start_1 = np.array([0, 1, 0, 0, 0, 0])
    start_2 = np.array([0, 0, 1, 0, 0, 0])
    start_3 = np.array([0, 0, 0, 1, 0, 0])
    weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])


    Z = util.generate_Z([.35, .35, .3], num_of_cells, rng)

    rates = rng.normal(loc=10, scale=1, size=num_of_cells)

    index_of_cells_cluster_1 = [index for index, value in enumerate(Z) if value == 1]
    index_of_cells_cluster_2 = [index for index, value in enumerate(Z) if value == 2]
    index_of_cells_cluster_3 = [index for index, value in enumerate(Z) if value == 3]

    rates_of_cluster_1 = rates[index_of_cells_cluster_1]
    rates_of_cluster_2 = rates[index_of_cells_cluster_2]
    rates_of_cluster_3 = rates[index_of_cells_cluster_3]

    num_of_states = len(weight_initial[0])

    var = 2

    random_state = np.random.RandomState(seed=0)
    C1, Y1, prob_C1 = util.generate_hmm_normal(rates_of_cluster_1, var, num_of_states, trans_1, start_1, seq_len, random_state)
    C2, Y2, prob_C2 = util.generate_hmm_normal(rates_of_cluster_2, var, num_of_states, trans_2, start_2, seq_len, random_state)
    C3, Y3, prob_C3 = util.generate_hmm_normal(rates_of_cluster_3, var, num_of_states, trans_3, start_3, seq_len, random_state)

    new_Y1 = np.swapaxes(Y1, 0, 1)
    new_Y2 = np.swapaxes(Y2, 0, 1)
    new_Y3 = np.swapaxes(Y3, 0, 1)
    data_sign = np.concatenate((new_Y1, new_Y2, new_Y3), axis=0)

    print("data splits:")
    print(len(new_Y1))
    print(len(new_Y2))

    C1[50:200] = 0
    means = rates_of_cluster_1 * .2
    data_sign[:len(new_Y1),50:200] = np.array([rng.normal(loc=mean, scale=math.sqrt(var), size=150) for mean in means])

    C1[0:26] = 0
    means = rates_of_cluster_1 * .2
    data_sign[:len(new_Y1),0:26] = np.array([rng.normal(loc=mean, scale=math.sqrt(var), size=26) for mean in means])

    plot(seq_len, data_sign, "CONF 9")
    label_0 = [0 for i in range(len(Y1[0]))]
    label_1 = [1 for j in range(len(Y2[0]))]
    label_2 = [2 for j in range(len(Y3[0]))]
    labels = np.concatenate((label_0, label_1, label_2))

    print("C1:")
    print(C1)
    print("Y1:")
    print(new_Y1)

    print("C2:")
    print(C2)
    print("Y2:")
    print(new_Y2)

    print("C3:")
    print(C3)
    print("Y3:")
    print(new_Y3)

    c_1 = [C1 for i in range(len(Y1[0]))]
    c_2 = [C2 for i in range(len(Y2[0]))]
    c_3 = [C3 for i in range(len(Y3[0]))]
    true_c = np.concatenate((c_1, c_2, c_3))

    data = data_sign

    # End of simulation

    def generate_categorical_prob(num_of_categories, random, alpha=None):
        if alpha is None:
            alpha = random.dirichlet([0.1] * num_of_categories)
        else:
            alpha = alpha * np.ones(num_of_categories)
        var = dirichlet.rvs(alpha=alpha, size=1, random_state=random)
        return var[0]

    def get_clustering_random(num_of_clusters, data, random):
        num_of_cells = len(data[:, 0])
        pi = np.zeros((num_of_cells, num_of_clusters))
        classes = np.zeros(num_of_cells)
        for n in range(num_of_cells):
            pi[n] = generate_categorical_prob(num_of_clusters, random)
            classes[n] = np.where(pi[n] == max(pi[n]))[0][0]
        return pi, classes

    ll = []
    new_pis = []
    new_weight_vertexes = []
    pis = []
    for r in range(11, 26):
        rng2 = np.random.default_rng(r)

        delta = np.array([10,10,10])
        delta_prior = np.array([1,1,1])
        theta = np.ones(num_of_cells)
        tau = np.ones(num_of_cells)
        theta_prior = 0
        tau_prior = 1
        alpha_prior = 0
        beta_prior = 0
        for n in range(num_of_cells):
            theta[n] = np.mean(data[n])  # mean of data # 10
            tau[n] = np.var(data[n])  # var of data # 1
        alpha_gam = 1
        beta_gam = np.var(data)
        weight_vertex = np.zeros((3, num_of_states, seq_len))
        weight_initial = np.ones((3, num_of_states)) / 3
        weight_edge = np.zeros((3, num_of_states, num_of_states))
        lam = np.zeros((3, num_of_states, num_of_states))
        lam_prior = np.ones((3, num_of_states, num_of_states))
        for k in range(3):
            for s in range(num_of_states):
                weight_edge[k, s] = generate_categorical_prob(num_of_states, rng2,.1)
                for l in range(seq_len):
                    weight_vertex[k, s, l] = rng2.uniform(0.1, 0.9)
            lam[k] = weight_edge[k] * 100 + .0000000000001
        pi, classes = get_clustering_random(3, data, rng2)

        prior = (delta, theta_prior, tau_prior, alpha_prior, beta_prior, lam)
        init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
        trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
        new_weight_vertex = vi(prior, init, data)
        ll = np.append(ll, vi.likelihood)
        new_pis = np.append(new_pis, new_pi)
        new_weight_vertexes = np.append(new_weight_vertexes, new_weight_vertex)
        pis = np.append(pis, pi)

    init_with_highest_ll = np.where(ll == np.max(ll))[0][0]
    print(ll[init_with_highest_ll])

    new_pis_reshaped = new_pis.reshape((15, num_of_cells, 2))
    new_weight_vertexes_reshaped = new_weight_vertexes.reshape((15, 2, 6, seq_len))
    pis_reshaped = pis.reshape((15, num_of_cells, 2))

    new_pi = new_pis_reshaped[init_with_highest_ll]
    new_weight_vertex = new_weight_vertexes_reshaped[init_with_highest_ll]
    pi = pis_reshaped[init_with_highest_ll]

    c = calculate_predicted_c(new_pi, new_weight_vertex, "CONF 9")

    print("TV distance:"+str(util.calculate_total_variation(c, true_c)))

    predicted_cluster = []
    for n in range(num_of_cells):
        cell = np.float64(pi[n])
        predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell)))
    v_measure_init = v_measure_score(labels, predicted_cluster)

    print("Init v-measure: " + str(v_measure_init))

    predicted_cluster = []
    for n in range(num_of_cells):
        cell = np.float64(new_pi[n])
        predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell)))
    v_measure = v_measure_score(labels, predicted_cluster)

    print("CopyMix v-measure: " + str(v_measure))

    return v_measure


if __name__ == "__main__":
    v_measures = []
    for seed in range(0, 30):
        clustering_performance = run(seed)
        v_measures = np.append(v_measures, clustering_performance)
    print('V-measures of 30 datasets: {0}'.format(v_measures))