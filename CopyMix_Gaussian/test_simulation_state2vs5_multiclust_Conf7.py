import numpy as np
import util, random
from CopyMix_Gaussian import inference
from sklearn.metrics.cluster import v_measure_score
from scipy.stats import dirichlet
import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def plot(seq_len, gc, name):
    i = 0
    for value in gc:
        if i < 63:
            color = 'r'
        elif i >= 63 and i < 123:
            color = 'b'
        else:
            color = 'green'
        plt.scatter(np.arange(seq_len), value, edgecolors=color, s=.3)
        plt.xlabel('sequence position')
        plt.ylabel('gc corrected ratio')
        plt.title(name)
        i += 1
    plt.savefig('/Users/negar/PycharmProjects/Test/CopyMix_Gaussian/' + name+'.png')


s = 2#12
rng = np.random.default_rng(s)
num_of_cells = 150
seq_len = 1000
trans_1 = np.array([[0, .2, .8, 0, 0, 0], [0, .1, .8, .1, 0, 0], [0, .1, .9, 0, 0, 0], [0, .1, .9, 0, 0, 0], [0, .1, .9, 0, 0, 0], [0, .1, .9, 0, 0, 0]])
#trans_1 = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#trans_1 = np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#trans_1 = np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, 0, .98, .01, .005, .005], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#trans_1 = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
trans_2 = np.array([[0, 0, 0, 0, .1, .9], [0, 0, 0, 0, .1, .9], [0, 0, 0, 0, .1, .9], [0, 0, 0, 0, .1, .9], [0, 0, 0, 0, .1, .9], [0, 0, 0, 0, .1, .9]])
#trans_2 = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])
#trans_2 = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
#trans_2 = np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0]])
#trans_2 = trans_1
trans_3 = trans_1#np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
# start_1 = np.array([0, 0, 1, 0, 0, 0])
# start_2 = np.array([0, 0, 0, 0, 0, 1])
# weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
#  start_1 = np.array([0, 1, 0, 0, 0, 0])
# start_2 = np.array([0, 1, 0, 0, 0, 0])
start_1 = np.array([0, 0, 1, 0, 0, 0])
start_2 = np.array([0, 0, 1, 0, 0, 0])
start_3 = np.array([0, 1, 0, 0, 0, 0])
weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

Z = util.generate_Z([.4, .4, .2], num_of_cells, rng)

rates = rng.normal(loc=10, scale=1, size=150)

index_of_cells_cluster_1 = [index for index, value in enumerate(Z) if value == 1]
index_of_cells_cluster_2 = [index for index, value in enumerate(Z) if value == 2]
index_of_cells_cluster_3 = [index for index, value in enumerate(Z) if value == 3]

rates_of_cluster_1 = rates[index_of_cells_cluster_1]
rates_of_cluster_2 = rates[index_of_cells_cluster_2]
rates_of_cluster_3 = rates[index_of_cells_cluster_3]

num_of_states = len(weight_initial[0])

var = 2

C1, Y1, prob_C1 = util.generate_hmm_normal(rates_of_cluster_1, var, num_of_states, trans_1, start_1, seq_len)
C2, Y2, prob_C2 = util.generate_hmm_normal(rates_of_cluster_2, var, num_of_states, trans_2, start_2, seq_len)
C3, Y3, prob_C3 = util.generate_hmm_normal(rates_of_cluster_3, var, num_of_states, trans_3, start_3, seq_len)

new_Y1 = np.swapaxes(Y1, 0, 1)
new_Y2 = np.swapaxes(Y2, 0, 1)
new_Y3 = np.swapaxes(Y3, 0, 1)
data_sign = np.concatenate((new_Y1, new_Y2, new_Y3), axis=0)
min_y = np.min(data_sign) + 1
# conf 7
# C2[10:20] = 3
# data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),10:20] = np.random.normal(loc=50, scale=math.sqrt(.1), size=10)[np.newaxis, :]
#
# C2[120:130] = 3
# data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),120:130] = np.random.normal(loc=50, scale=math.sqrt(.1), size=10)[np.newaxis, :]

# C2[600:750] = 5
# data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),600:750] = np.random.normal(loc=50, scale=math.sqrt(.1), size=150)[np.newaxis, :]

# C3[10:20] = 1
# data_sign[len(new_Y1)+len(new_Y2):,10:20] = min_y * 1/2
#
# C3[100:150] = 1
# data_sign[len(new_Y1)+len(new_Y2):,100:150] = min_y * 1/2

C3[500:700] = 4
data_sign[len(new_Y1)+len(new_Y2):,500:700] = np.random.normal(loc=90, scale=math.sqrt(.3), size=200)[np.newaxis, :]

C3[800:1000] = 4
data_sign[len(new_Y1)+len(new_Y2):,800:1000] = np.random.normal(loc=90, scale=math.sqrt(.3), size=200)[np.newaxis, :]

# C3[500:550] = 0
# data_sign[len(new_Y1)+len(new_Y2):,500:550] = 1/2

# C3[550:650] = 4
# data_sign[len(new_Y1)+len(new_Y2):,550:650] = min_y * 8

plot(seq_len, data_sign, "CONF 7")
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

def get_clustering_kmeans(num_of_clusters, data):
    num_of_cells = len(data[:, 0])
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0).fit(data)
    classes = kmeans.labels_
    pi = np.zeros((num_of_cells, num_of_clusters))
    for n in range(num_of_cells):
        pi[n, classes[n]] = 1
    return pi, classes


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
        pi[n] = generate_categorical_prob(num_of_clusters) # generate_categorical_prob(num_of_clusters, 10)
        classes[n] = np.where(pi[n] == max(pi[n]))[0][0]
    return pi, classes

delta = np.array([10,10,5])
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
alpha_gam = 1 #.01 1
beta_gam = np.var(data) #.01   2
weight_vertex = np.zeros((3, num_of_states, seq_len))
weight_initial = np.ones((3, num_of_states)) / 3
weight_edge = np.zeros((3, num_of_states, num_of_states))
lam = np.zeros((3, num_of_states, num_of_states))
lam_prior = np.ones((3, num_of_states, num_of_states))
for k in range(3):
    for s in range(num_of_states):
        weight_edge[k, s] = generate_categorical_prob(num_of_states, .5)
        for l in range(seq_len):
            weight_vertex[k, s, l] = rng.uniform(0.1, 0.9)
    lam[k] = weight_edge[k] * 100 + .0000000000001
pi, classes = get_clustering_random(3, data) # get_clustering_kmeans(3, data) #

lam[0] = weight_edge[0] * 100 + .0000000000001
lam[1] = weight_edge[1] * 100 + .0000000000001
lam[2] = weight_edge[2] * 100 + .0000000000001
#
# pi = np.array([[0, 0, 1], [0, 0, 1],
#                [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
#                [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]
#                 , [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
#                [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
#                 [0, 0, 1], [0, 0, 1], [0, 0, 1],
#                [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
#                [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
#                [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
#                [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
#                [1, 0, 0], [1, 0, 0],
#                [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
#                [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
#                [0, 0, 1]
#                ])

# pi = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
#                #[1, 0], [1, 0], [1, 0], [1, 0],
#                #[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]
#                 , [1, 0], [1, 0], [1, 0],
#                [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
#                [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
#                [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
#                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
#                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
#                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], ])
# # #

prior = (delta_prior, theta_prior, tau_prior, alpha_prior, beta_prior, lam_prior)
init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
new_weight_vertex = inference.vi(prior, init, data)

print(new_pi)

print(new_alpha_gam)
print(new_beta_gam)
print(rates_of_cluster_2)
print(new_theta)
print(new_tau)

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