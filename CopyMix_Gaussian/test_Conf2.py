import numpy as np
import util
import inference
from sklearn.metrics.cluster import v_measure_score
from scipy.stats import dirichlet
from matplotlib import pyplot as plt
import math


def calculate_predicted_c(pi, weight_vertex, name):
    K = pi.shape[1]
    M = len(weight_vertex[0, 0, :])
    predicted_c = np.zeros((K, M))
    fig, ax = plt.subplots()
    for i in range(K):
        predicted_c[i] = [np.argmax(weight_vertex[i, :, m]) for m in range(M)]
        if i == 0:
            color = 'r'
            m = "v"
        elif i == 1:
            color = 'b'
            m = "."
        else:
            color = 'green'
            m = "*"

        ax.scatter(np.arange(seq_len), predicted_c[i], edgecolors=color, s=10, marker=m, alpha=.3)
        ax.set_xlabel('sequence position')
        ax.set_ylabel('copy number')
        ax.set_title(name + "_estimated_copy_number")
    plt.savefig('./plots/'+name + "_estimated_copy_number.png")
    return predicted_c


def plot(seq_len, gc, name, locs):
    fig, ax = plt.subplots()
    i = 0
    for value in gc:
        if i < 74:
            color = 'r'
        else:
            color = 'b'
        ax.scatter(np.arange(seq_len), value, edgecolors=color, s=.3)
        ax.set_xticks(locs[:-1])
        ax.set_xlabel('sequence position')
        ax.set_ylabel('gc corrected ratio')
        ax.set_title(name)
        i += 1
    plt.savefig('./plots/CONF 2.png')


s = 12
rng = np.random.default_rng(s)
num_of_cells = 150
seq_len = 200
locs = util.get_chrom_locations(seq_len)

trans_1 = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
trans_2 = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
start_1 = np.array([0, 0, 1, 0, 0, 0])
start_2 = np.array([0, 0, 1, 0, 0, 0])
weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

Z = util.generate_Z([1/2, 1/2], num_of_cells, rng)

rates = rng.normal(loc=10, scale=1, size=150)

index_of_cells_cluster_1 = [index for index, value in enumerate(Z) if value == 1]
index_of_cells_cluster_2 = [index for index, value in enumerate(Z) if value == 2]

rates_of_cluster_1 = rates[index_of_cells_cluster_1]
rates_of_cluster_2 = rates[index_of_cells_cluster_2]

num_of_states = len(weight_initial[0])

var = 2

random_state = np.random.RandomState(seed=0)
C1, Y1, prob_C1 = util.generate_hmm_normal(rates_of_cluster_1, var, num_of_states, trans_1, start_1, seq_len, random_state)
C2, Y2, prob_C2 = util.generate_hmm_normal(rates_of_cluster_2, var, num_of_states, trans_2, start_2, seq_len, random_state)

new_Y1 = np.swapaxes(Y1, 0, 1)
new_Y2 = np.swapaxes(Y2, 0, 1)
data_sign = np.concatenate((new_Y1, new_Y2), axis=0)

print("data splits:")
print(len(new_Y1))
print(len(new_Y2))

C2[10:30] = 2
means = rates_of_cluster_2 * 2
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),10:30] = np.array([rng.normal(loc=mean, scale=math.sqrt(var), size=20) for mean in means])

plot(seq_len, data_sign, "CONF 2", locs)
label_0 = [0 for i in range(len(Y1[0]))]
label_1 = [1 for j in range(len(Y2[0]))]
labels = np.concatenate((label_0, label_1))

print("C1:")
print(C1)
print("Y1:")
print(new_Y1)

print("C2:")
print(C2)
print("Y2:")
print(new_Y2)

c_1 = [C1 for i in range(len(Y1[0]))]
c_2 = [C2 for i in range(len(Y2[0]))]
true_c = np.concatenate((c_1, c_2))
data = data_sign

# End of simulation


def generate_categorical_prob(num_of_categories, alpha=None):
    if alpha is None:
        alpha = rng.dirichlet([0.1] * num_of_categories)
    else:
        alpha = alpha * np.ones(num_of_categories)
    var = dirichlet.rvs(alpha=alpha, size=1, random_state=rng)
    return var[0]


def get_clustering_random(num_of_clusters, data):
    num_of_cells = len(data[:, 0])
    pi = np.zeros((num_of_cells, num_of_clusters))
    classes = np.zeros(num_of_cells)
    for n in range(num_of_cells):
        pi[n] = generate_categorical_prob(num_of_clusters)
        classes[n] = np.where(pi[n] == max(pi[n]))[0][0]
    return pi, classes

delta = np.array([10,10])
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
weight_vertex = np.zeros((2, num_of_states, seq_len))
weight_initial = np.ones((2, num_of_states)) / 2
weight_edge = np.zeros((2, num_of_states, num_of_states))
lam = np.zeros((2, num_of_states, num_of_states))
for k in range(2):
    for s in range(num_of_states):
        weight_edge[k, s] = generate_categorical_prob(num_of_states, .1)
        for l in range(seq_len):
            weight_vertex[k, s, l] = rng.uniform(0.1, 0.9)
    lam[k] = weight_edge[k] * 100 + .0000000000001
pi, classes = get_clustering_random(2, data)

prior = (delta, theta_prior, tau_prior, alpha_prior, beta_prior, lam)
init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
new_weight_vertex = inference.vi(locs, prior, init, data)

c = calculate_predicted_c(new_pi, new_weight_vertex, "CONF 2")
print(c[0])
print('#####')
print(c[1])

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