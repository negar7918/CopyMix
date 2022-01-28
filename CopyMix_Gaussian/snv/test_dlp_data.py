import numpy as np
import csv
from sklearn.metrics.cluster import v_measure_score
import inference_with_snv as inference
import matplotlib.pylab as plt
from scipy.stats import dirichlet

rng = np.random.default_rng(32)


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


# SNV emissions
p_X = np.load("./log_likelihood_tensor_v7.npy")

dlp_labels = np.empty((0, 1), dtype=object)
with open('../dlp_labels_removed_outlier_cells.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        dlp_labels = np.vstack((dlp_labels, row))

dlp_l = dlp_labels.flatten()

input = np.empty((0, 6206), dtype=object)
with open('./ov2295_corrected_ratios_removed_outlier_cells.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        input = np.vstack((input, row))

data = np.array(input, dtype=float)


num_of_states = 6
num_of_cells = 891
seq_len = 6206
genome_len = p_X.shape[1]
num_of_clusters = 9

delta = np.ones(num_of_clusters) * 10
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
weight_vertex = np.zeros((num_of_clusters, num_of_states, seq_len))
weight_initial = np.ones((num_of_clusters, num_of_states)) / num_of_clusters
weight_edge = np.zeros((num_of_clusters, num_of_states, num_of_states))
lam = np.zeros((num_of_clusters, num_of_states, num_of_states))
for k in range(num_of_clusters):
    for s in range(num_of_states):
        weight_edge[k, s] = generate_categorical_prob(num_of_states, 10)
        for l in range(seq_len):
            weight_vertex[k, s, l] = rng.uniform(.1, .9)
    lam[k] = weight_edge[k] * 100 + .0000000000001
pi, classes = get_clustering_random(num_of_clusters, data)

epsilon, omega, gamma, eta = .9, 1, np.ones(num_of_clusters), np.ones(num_of_clusters)
epsilon_prior, omega_prior, gamma_prior, eta_prior = 2, 2, 1 ,1
sigma = np.ones((num_of_clusters, genome_len)) * .5

###### read from best init in Gaussian CopyMix
import pickle
f = open('../50inits/inits_dlp_approx_6states_9clust_25it__12', 'rb')
inits = pickle.load(f)
f.close()

d = inits.reshape(1, 23)

trans, delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex,new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, new_weight_edge, new_weight_vertex,method, likelihood, num_of_clusters = d[0]
######

prior = (delta, theta_prior, tau_prior, alpha_prior, beta_prior, lam, gamma_prior, eta_prior)
init = (new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge, new_weight_vertex, gamma, eta, sigma)
print("starting VI")
trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge,\
new_weight_vertex, new_gamma, new_eta, new_sigma = inference.vi(prior, init, data, p_X)


predicted_cluster = []
for n in range(num_of_cells):
    cell = np.float64(new_pi[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell)))

estimated_num_clust = len(np.unique(predicted_cluster))
print('*****************************')
print(estimated_num_clust)

file = open("./result.txt", "w+")
 
# Saving the array in a text file
content = str(np.array([estimated_num_clust, trans, new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, weight_initial, new_weight_edge,\
        new_weight_vertex, new_gamma, new_eta, new_sigma]))
file.write(content)
file.close()

predicted_cluster_init = []
for n in range(num_of_cells):
   cell = np.float64(pi[n])
   predicted_cluster_init = np.append(predicted_cluster_init, np.where(cell == max(cell)))
v_measure_init = v_measure_score(dlp_l, predicted_cluster_init)

print("Init v-measure: " + str(v_measure_init))

predicted_cluster = []
for n in range(num_of_cells):
    cell = np.float64(new_pi[n])
    predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell)))
v_measure = v_measure_score(dlp_l, predicted_cluster)

print("CopyMix v-measure: " + str(v_measure))


def cocluster(predicted_cluster, num_of_cells, dlp_l):
    heatmap_clustering = np.zeros((9, 9))
    i = 0
    l = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    for label in l:
        for ind in range(num_of_cells):
            if predicted_cluster[ind] == label and dlp_l[ind] == 'E':
                heatmap_clustering[i, 0] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'F':
                heatmap_clustering[i, 1] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'G':
                heatmap_clustering[i, 2] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'H':
                heatmap_clustering[i, 3] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'I':
                heatmap_clustering[i, 4] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'A':
                heatmap_clustering[i, 5] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'B':
                heatmap_clustering[i, 6] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'C':
                heatmap_clustering[i, 7] += 1
            elif predicted_cluster[ind] == label and dlp_l[ind] == 'D':
                heatmap_clustering[i, 8] += 1

        i += 1

    plt.imshow(heatmap_clustering)
    plt.xlabel('9 clones')
    plt.ylabel('9 clones')
    plt.title('SNV-CopyMix co-clustering')
    plt.colorbar()
    plt.show()

    print(heatmap_clustering)
    print('\n')


cocluster(predicted_cluster, num_of_cells, dlp_l)



