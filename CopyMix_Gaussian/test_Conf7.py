import numpy as np
import util
import inference
from sklearn.metrics.cluster import v_measure_score
from scipy.stats import dirichlet
import math
from matplotlib import pyplot as plt


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
        if i < 60:
            color = 'r'
        elif i >= 60 and i < 107:
            color = 'b'
        else:
            color = 'green'
        ax.scatter(np.arange(seq_len), value, edgecolors=color, s=.3)
        ax.set_xticks(locs[:-1])
        ax.set_xlabel('sequence position')
        ax.set_ylabel('gc corrected ratio')
        ax.set_title(name)
        i += 1
    plt.savefig('./plots/' + name+'.png')


s = 12
rng = np.random.default_rng(s)
num_of_cells = 150
seq_len = 200
trans_1 = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
trans_2 = np.array([[.9, .1, 0, 0, 0, 0], [0, .9, .1, 0, 0, 0], [0, 0, .9, .1, 0, 0], [0, 0, 0, .9, 0, .1], [0, 0, 0, 0, .9, .1], [0, 0, 0, 0, .1, .9]])
trans_3 = np.array([[.9, .1, 0, 0, 0, 0], [0, .9, .1, 0, 0, 0], [.9, .1, 0, 0, 0, 0], [0, .9, 0, 0, 0, .1], [.9, 0, 0, 0, 0, .1], [0, 0, .1, 0, 0, .9]])
start_1 = np.array([0, 0, 0, 0, 0, 1])
start_2 = np.array([0, 0, .5, .5, 0, 0])
start_3 = np.array([0, 0, 0, 1, 0, 0])
weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

locs = util.get_chrom_locations(seq_len)

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

plot(seq_len, data_sign, "CONF 10", locs)
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

#s = 21 -67531 #121 -89706 #3 -88515 #2 -73086 77% #1 -73993 76% #0 -89587 #12 -73681 76% #17 -88425 74% #19 -90176  # for seed 0
#s = 21 76042 #17 -80457 71% #19 75817 71% # for seed 1
#s = 21 -65307 #17 -73590 74% # for seed 2
#s = 21 -70281 #17 -76958 72% # for seed 3
#s = 14 -66434 #3 -74713 60% #0 -89575 #1 -82525 75% #2 -73882 58% #12 -74711 60% #21 -89366 74% #17 -87827 75% # for seed 4
#s = 14 -71068 #17 -76085 73% # for seed 5
#s = 14 -64199 #17 -74534 72% # for seed 6
#s = 21 -65647 #14 -115013 #17 -73847 74% # for seed 7
#s = 21 -68226 #17 -75001 71% # for seed 8
#s = 21 -67247 #17 -73748 76% # for seed 9
#s = 21 -68826 #17 -74508 73% # for seed 10
#s = 14 -62873 #21 -87531 #17 -87067 73% # for seed 11
#s = 21 -69112 #17 -85189 # for seed 12
#s = 21 -68160 #17 -76407 70% # for seed 13
#s = 21 -68988 #17 -85238 # for seed 14
#s = 17 -66128 #21 -77242 # for seed 15
#s = 14 -61981 #21 -75200 71% #17 -75202 71% # for seed 16
#s = 12 -70889 69% #23 -75333 69% #13 -75352 #19 -83273 #14 -76314 #21 -75375 #17 -75345 # for seed 17
#s = 12 -74777 65% #14 -80749 #21 -80931 #17 -86945 # for seed 18
#s = 21 -70681 #17 -76142 # for seed 19
#s = 25 -62659 #19 -71439 #12 -71395 75% #14 -73769 #21 -73588 #17 -71399 74% # for seed 20
#s = 21 -68599 #17 -75772 73% # for seed 21
#s = 21 -69565 #17 -75666 # for seed 22
#s = 21 -68997 # 17 -87793 # for seed 23
#s = 25 -74339 #21 -74347 #17 -83195 # for seed 24
#s = 25 -70508 72% #12 -70516 #14 -72885 #21 -85512 #17 -77199 # for seed 25
#s = 21 -66540 #17 -74099 73% # for seed 26
#s = 12 -70989 75% #19 -73623 #25 -73592 #14 -73516 75% #21 -77154 #17 -73562 75% # for seed 27
#s = 21 -63204 #17 -105183 # for seed 28
#s = 21 -72360 #17 -85603 # for seed 29
#rng2 = np.random.default_rng(s)


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
        weight_edge[k, s] = generate_categorical_prob(num_of_states, .1)
        for l in range(seq_len):
            weight_vertex[k, s, l] = rng.uniform(0.1, 0.9)
    lam[k] = weight_edge[k] * 100 + .0000000000001
pi, classes = get_clustering_random(3, data)

prior = (delta_prior, theta_prior, tau_prior, alpha_prior, beta_prior, lam_prior)
init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
new_weight_vertex = inference.vi(locs, prior, init, data)

c = calculate_predicted_c(new_pi, new_weight_vertex, "CONF 10")
print(c[0])
print('#####')
print(c[1])
print('#####')
print(c[2])

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