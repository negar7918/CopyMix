import numpy as np
import util, random
from CopyMix_Gaussian import inference
from sklearn.metrics.cluster import v_measure_score
from scipy.stats import dirichlet
import math
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


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
        elif i == 2:
            color = 'green'
            m = "*"
        else:
            color = 'black'
            m = "s"

        ax.scatter(np.arange(seq_len), predicted_c[i], edgecolors=color, s=10, marker=m, alpha=.3)
        ax.set_xlabel('sequence position')
        ax.set_ylabel('copy number')
        ax.set_title(name + "_estimated_copy_number")
    plt.savefig(name + "_estimated_copy_number.png")
    return predicted_c


def plot(seq_len, gc, name):
    i = 0
    for value in gc:
        if i < 56:
            color = 'r'
        elif i >= 56 and i < 101:
            color = 'b'
        elif i >= 101 and i < 145:
            color = 'green'
        else:
            color = 'black'
        plt.scatter(np.arange(seq_len), value, edgecolors=color, s=.3)
        plt.xlabel('sequence position')
        plt.ylabel('gc corrected ratio')
        plt.title(name)
        i += 1
    plt.savefig('/Users/negar/PycharmProjects/Test/CopyMix_Gaussian/'+name+'.png')


s = 12
rng = np.random.default_rng(s)
num_of_cells = 200
seq_len = 150
trans_1 = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
#trans_1 = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
trans_2 = trans_1 #np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#trans_1 = np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, 0, .98, .01, .005, .005], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#trans_1 = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
#trans_4 = np.array([[0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])
#trans_2 = trans_1#np.array([[.7, .2, 0, 0, 0, .1], [0, .7, .2, 0, 0, .1], [0, 0, .7, .2, .1, 0], [0, 0, 0, .9, 0, .1], [0, 0, 0, 0, .9, .1], [0, 0, 0, 0, .1, .9]])
#trans_2 = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
#trans_2 = np.array([[0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0], [0, .98, .02, 0, 0, 0]])
#trans_2 = trans_1
trans_3 = trans_1 #np.array([[0, .1, .9, 0, 0, 0], [0, 0, .92, .08, 0, 0], [.5, 0, .5, 0, 0, 0], [0, 0, 0, .6, .4, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
trans_4 = trans_1
start_1 = np.array([0, 1, 0, 0, 0, 0])
start_2 = np.array([0, 0, 1, 0, 0, 0])
start_3 = np.array([0, 1, 0, 0, 0, 0])
start_4 = np.array([0, 0, 1, 0, 0, 0])
weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

Z = util.generate_Z([1/4, 1/4, 1/4, 1/4], num_of_cells, rng)

rates = rng.normal(loc=10, scale=1, size=200) #np.ones(num_of_cells) * 10

index_of_cells_cluster_1 = [index for index, value in enumerate(Z) if value == 1]
index_of_cells_cluster_2 = [index for index, value in enumerate(Z) if value == 2]
index_of_cells_cluster_3 = [index for index, value in enumerate(Z) if value == 3]
index_of_cells_cluster_4 = [index for index, value in enumerate(Z) if value == 4]

rates_of_cluster_1 = rates[index_of_cells_cluster_1]
rates_of_cluster_2 = rates[index_of_cells_cluster_2]
rates_of_cluster_3 = rates[index_of_cells_cluster_3]
rates_of_cluster_4 = rates[index_of_cells_cluster_4]

num_of_states = len(weight_initial[0])

var = 2

C1, Y1, prob_C1 = util.generate_hmm_normal(rates_of_cluster_1, var, num_of_states, trans_1, start_1, seq_len)
C2, Y2, prob_C2 = util.generate_hmm_normal(rates_of_cluster_2, var, num_of_states, trans_2, start_2, seq_len)
C3, Y3, prob_C3 = util.generate_hmm_normal(rates_of_cluster_3, var, num_of_states, trans_3, start_3, seq_len)
C4, Y4, prob_C4 = util.generate_hmm_normal(rates_of_cluster_4, var, num_of_states, trans_4, start_4, seq_len)

new_Y1 = np.swapaxes(Y1, 0, 1)
new_Y2 = np.swapaxes(Y2, 0, 1)
new_Y3 = np.swapaxes(Y3, 0, 1)
new_Y4 = np.swapaxes(Y4, 0, 1)

data_sign = np.concatenate((new_Y1, new_Y2, new_Y3, new_Y4), axis=0)
min_y = np.min(data_sign) + 1

C2[30:50] = 2
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),30:50] = rng.normal(loc=50, scale=math.sqrt(.1), size=20)[np.newaxis, :]

C2[50:70] = 0
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),50:70] = rng.normal(loc=1, scale=math.sqrt(.1), size=20)[np.newaxis, :]


C3[10:30] = 4
data_sign[len(new_Y1)+len(new_Y2):len(new_Y1)+len(new_Y2)+len(new_Y3),10:30] = rng.normal(loc=90, scale=math.sqrt(.3), size=20)[np.newaxis, :]


C4[100:130] = 2
data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):,100:130] = rng.normal(loc=50, scale=math.sqrt(.1), size=30)[np.newaxis, :]

C4[130:150] = 4
data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):,130:150] = rng.normal(loc=90, scale=math.sqrt(.3), size=20)[np.newaxis, :]


plot(seq_len, data_sign, "CONF 14")
label_0 = [0 for i in range(len(Y1[0]))]
label_1 = [1 for j in range(len(Y2[0]))]
label_2 = [2 for j in range(len(Y3[0]))]
label_3 = [3 for j in range(len(Y4[0]))]
labels = np.concatenate((label_0, label_1, label_2, label_3))

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

print("C4:")
print(C4)
print("Y4:")
print(new_Y4)

c_1 = [C1 for i in range(len(Y1[0]))]
c_2 = [C2 for i in range(len(Y2[0]))]
c_3 = [C3 for i in range(len(Y3[0]))]
c_4 = [C4 for i in range(len(Y4[0]))]
true_c = np.concatenate((c_1, c_2, c_3, c_4))

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
        pi[n] = generate_categorical_prob(num_of_clusters)
        classes[n] = np.where(pi[n] == max(pi[n]))[0][0]
    return pi, classes

delta = np.array([10,10,10,5])
theta = np.ones(num_of_cells)
tau = np.ones(num_of_cells)
theta_prior = 0
tau_prior = 1
alpha_prior = 0
beta_prior = 0
for n in range(num_of_cells):
    theta[n] = np.mean(data[n])  # mean of data # 10
    tau[n] =  np.var(data[n])  # var of data # 1
alpha_gam = 1 #.01 1
beta_gam = np.var(data) #.01   2
weight_vertex = np.zeros((4, num_of_states, seq_len))
weight_initial = np.ones((4, num_of_states)) / 4
weight_edge = np.ones((4, num_of_states, num_of_states)) / num_of_states
lam = np.zeros((4, num_of_states, num_of_states))
for k in range(4):
    for s in range(num_of_states):
        weight_edge[k, s] = generate_categorical_prob(num_of_states, .5)
        for l in range(seq_len):
            weight_vertex[k, s, l] = rng.uniform(0.1, 0.9)
    lam[k] = weight_edge[k] * 100 + .0000000000001
pi, classes = get_clustering_random(4, data)

#pi[0, :] = pi[-1, :]
#pi[1, :] = pi[-2, :]
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

prior = (delta, theta_prior, tau_prior, alpha_prior, beta_prior, lam)
init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
new_weight_vertex = inference.vi(prior, init, data)

print(new_pi)

print(new_weight_edge)

#c = calculate_predicted_c(new_pi, new_weight_vertex, "CONF 14")

# print(c[0])
# print(c[1])
# print(c[2])
# print(c[3])
# print(new_alpha_gam)
# print(new_beta_gam)
# print(rates_of_cluster_2)
# print(new_theta)
# print(new_tau)

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