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
        if i < 36:
            color = 'r'
        elif i >= 36 and i < 78:
            color = 'b'
        elif i >= 78 and i < 107:
            color = 'green'
        elif i >= 107 and i < 128:
            color = 'brown'
        else:
            color = 'pink'
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
trans_3 = trans_1
trans_2 = trans_1
trans_4 = trans_1
start_1 = np.array([0, 1, 0, 0, 0, 0])
start_2 = np.array([0, 0, 1, 0, 0, 0])
start_3 = np.array([0, 0, 0, 1, 0, 0])
start_4 = np.array([0, 0, 0, 0, 0, 1])
start_5 = np.array([0, 0, 0, 0, 1, 0])
weight_initial = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]])

trans_1 = np.array([[0, .2, .8, 0, 0, 0], [0, .1, .8, .1, 0, 0], [0, .1, .9, 0, 0, 0], [0, .1, .9, 0, 0, 0], [0, .1, .9, 0, 0, 0], [0, .1, .9, 0, 0, 0]])
trans_5 = trans_1

locs = util.get_chrom_locations(seq_len)

Z = util.generate_Z([.2, .2, .2, .2, .2], num_of_cells, rng)

rates = rng.normal(loc=10, scale=.1, size=num_of_cells)

index_of_cells_cluster_1 = [index for index, value in enumerate(Z) if value == 1]
index_of_cells_cluster_2 = [index for index, value in enumerate(Z) if value == 2]
index_of_cells_cluster_3 = [index for index, value in enumerate(Z) if value == 3]
index_of_cells_cluster_4 = [index for index, value in enumerate(Z) if value == 4]
index_of_cells_cluster_5 = [index for index, value in enumerate(Z) if value == 5]

rates_of_cluster_1 = rates[index_of_cells_cluster_1]
rates_of_cluster_2 = rates[index_of_cells_cluster_2]
rates_of_cluster_3 = rates[index_of_cells_cluster_3]
rates_of_cluster_4 = rates[index_of_cells_cluster_4]
rates_of_cluster_5 = rates[index_of_cells_cluster_5]

num_of_states = len(weight_initial[0])

var = 2

random_state = np.random.RandomState(seed=0)
C1, Y1, prob_C1 = util.generate_hmm_normal(rates_of_cluster_1, var, num_of_states, trans_1, start_1, seq_len, random_state)
C2, Y2, prob_C2 = util.generate_hmm_normal(rates_of_cluster_2, var, num_of_states, trans_2, start_2, seq_len, random_state)
C3, Y3, prob_C3 = util.generate_hmm_normal(rates_of_cluster_3, var, num_of_states, trans_3, start_3, seq_len, random_state)
C4, Y4, prob_C4 = util.generate_hmm_normal(rates_of_cluster_4, var, num_of_states, trans_4, start_4, seq_len, random_state)
C5, Y5, prob_C5 = util.generate_hmm_normal(rates_of_cluster_5, var, num_of_states, trans_5, start_5, seq_len, random_state)

new_Y1 = np.swapaxes(Y1, 0, 1)
new_Y2 = np.swapaxes(Y2, 0, 1)
new_Y3 = np.swapaxes(Y3, 0, 1)
new_Y4 = np.swapaxes(Y4, 0, 1)
new_Y5 = np.swapaxes(Y5, 0, 1)
data_sign = np.concatenate((new_Y1, new_Y2, new_Y3, new_Y4, new_Y5), axis=0)

print("data splits:")
print(len(new_Y1))
print(len(new_Y2))
print(len(new_Y3))
print(len(new_Y4))

C1[30:40] = 5
means = rates_of_cluster_1
data_sign[:len(new_Y1),30:40] = np.array([rng.normal(loc=70, scale=math.sqrt(var), size=10) for mean in means])
C1[70:80] = 5
means = rates_of_cluster_1
data_sign[:len(new_Y1),70:80] = np.array([rng.normal(loc=70, scale=math.sqrt(var), size=10) for mean in means])
C1[120:130] = 5
means = rates_of_cluster_1
data_sign[:len(new_Y1),120:130] = np.array([rng.normal(loc=70, scale=math.sqrt(var), size=10) for mean in means])


C2[70:90] = 2
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),70:90] = rng.normal(loc=40, scale=math.sqrt(var), size=20)[np.newaxis, :]
C2[90:110] = 3
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),90:110] = rng.normal(loc=50, scale=math.sqrt(10), size=20)[np.newaxis, :]
C2[110:130] = 2
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),110:130] = rng.normal(loc=40, scale=math.sqrt(var), size=20)[np.newaxis, :]
C2[150:170] = 4
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),150:170] = rng.normal(loc=60, scale=math.sqrt(5), size=20)[np.newaxis, :]
C2[170:190] = 5
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),170:190] = rng.normal(loc=70, scale=math.sqrt(10), size=20)[np.newaxis, :]
C2[190:200] = 4
data_sign[len(new_Y1):len(new_Y1)+len(new_Y2),190:200] = rng.normal(loc=60, scale=math.sqrt(10), size=10)[np.newaxis, :]


C3[10:26] = 0
data_sign[len(new_Y1)+len(new_Y2):len(new_Y1)+len(new_Y2)+len(new_Y3),10:26] = rng.normal(loc=.5, scale=math.sqrt(.1), size=16)[np.newaxis, :]

C3[140:170] = 2
data_sign[len(new_Y1)+len(new_Y2):len(new_Y1)+len(new_Y2)+len(new_Y3),140:170] = rng.normal(loc=40, scale=math.sqrt(var), size=30)[np.newaxis, :]

C4[10:20] = 3
means = rates_of_cluster_4
data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):len(new_Y1)+len(new_Y2)+len(new_Y3)+len(new_Y4),10:20] = np.array([rng.normal(loc=50, scale=math.sqrt(var), size=10) for mean in means])

C4[20:40] = 2
means = rates_of_cluster_4
data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):len(new_Y1)+len(new_Y2)+len(new_Y3)+len(new_Y4),20:40] = np.array([rng.normal(loc=30, scale=math.sqrt(var), size=20) for mean in means])

C4[150:170] = 0
means = rates_of_cluster_4
data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):len(new_Y1)+len(new_Y2)+len(new_Y3)+len(new_Y4),150:170] = np.array([rng.normal(loc=1, scale=math.sqrt(var), size=20) for mean in means])


data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):len(new_Y1)+len(new_Y2)+len(new_Y3)+len(new_Y4), 0] = 1
for m in map(int, locs[13:-1]):
    data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):len(new_Y1)+len(new_Y2)+len(new_Y3)+len(new_Y4), m+1] = data_sign[len(new_Y1)+len(new_Y2)+len(new_Y3):len(new_Y1)+len(new_Y2)+len(new_Y3)+len(new_Y4), 0]


plot(seq_len, data_sign, "CONF 16", locs)
label_0 = [0 for i in range(len(Y1[0]))]
label_1 = [1 for j in range(len(Y2[0]))]
label_2 = [2 for j in range(len(Y3[0]))]
label_3 = [3 for j in range(len(Y4[0]))]
label_4 = [4 for j in range(len(Y5[0]))]
labels = np.concatenate((label_0, label_1, label_2, label_3, label_4))

c_1 = [C1 for i in range(len(Y1[0]))]
c_2 = [C2 for i in range(len(Y2[0]))]
c_3 = [C3 for i in range(len(Y3[0]))]
c_4 = [C4 for i in range(len(Y4[0]))]
c_5 = [C5 for i in range(len(Y5[0]))]
true_c = np.concatenate((c_1, c_2, c_3, c_4, c_5))

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


delta = np.array([10,10,10,10,20])
delta_prior = np.array([1,1,1,1,1])
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
weight_vertex = np.zeros((5, num_of_states, seq_len))
weight_initial = np.ones((5, num_of_states)) / 5
weight_edge = np.zeros((5, num_of_states, num_of_states))
lam = np.zeros((5, num_of_states, num_of_states))
lam_prior = np.ones((5, num_of_states, num_of_states))
for k in range(5):
    for s in range(num_of_states):
        weight_edge[k, s] = generate_categorical_prob(num_of_states, .1)
        for l in range(seq_len):
            weight_vertex[k, s, l] = rng.uniform(0.1, 0.9)
    lam[k] = weight_edge[k] * 100 + .0000000000001
pi, classes = get_clustering_random(5, data)

prior = (delta_prior, theta_prior, tau_prior, alpha_prior, beta_prior, lam_prior)
init = (delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex)
trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, \
new_weight_vertex = inference.vi(locs, prior, init, data)

c = calculate_predicted_c(new_pi, new_weight_vertex, "CONF 16")
print(c[0])
print('#####')
print(c[1])
print('#####')
print(c[2])
print('#####')
print(c[3])
print('#####')
print(c[4])

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