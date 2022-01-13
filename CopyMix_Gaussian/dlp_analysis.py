import numpy as np
import csv
import scipy.stats as stats
from scipy.stats.distributions import chi2
import pickle
from sklearn.metrics.cluster import v_measure_score


def calculate_mean_std(cells):
    return np.mean(cells), np.std(cells)


def calculate_ll(cells, labels):
    ll = 0
    l = np.unique(labels)
    for label in l:
        indexes = np.where(labels == label)[0]
        data = cells[indexes]
        mean, std = calculate_mean_std(data)
        dist = stats.norm(mean, std)
        for cell in data:
            ll += dist.logpdf(cell)
    return np.sum(ll)


def likelihood_ratio(cells, cluster_labels_copymix, cluster_labels_dlp):
    h1 = calculate_ll(cells, cluster_labels_dlp)
    h0 = calculate_ll(cells, cluster_labels_copymix) # null hypothesis with 4 clusters/degree-of-freedom
    print('LR value: ' + str(h1) + ' '+str(h0))
    return 2 * (h0 - h1)


def LR_test(cells, cluster_labels_copymix, cluster_labels_dlp):
    lr = likelihood_ratio(cells, cluster_labels_copymix, cluster_labels_dlp)
    p = chi2.sf(lr, 9 - 4) # difference of degrees of freedom is 9-4 = 5 clusters
    print('p: %.30f' % p)


dlp_labels = np.empty((0, 1), dtype=object)
with open('./dlp_labels_removed_outlier_cells.csv', encoding='US-ASCII') as f:
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
l = 6206

id = 'inits_dlp_approx_6states_9clust_25it__'
num_of_clusters_DIC = 9

lls = []
inits = np.empty((0, 23), dtype=object)
for i in range(1, 51):
    f = open('./50inits/' + id  + str(i), 'rb')
    d = pickle.load(f)
    inits = np.vstack((inits, d))
    lls = np.append(lls, d[-2])
    f.close()

init_with_highest_ll = np.where(lls == np.max(lls))[0][0]

trans, delta, theta, tau, alpha_gam, beta_gam, lam, pi, weight_initial, weight_edge, weight_vertex,\
new_delta, new_theta, new_tau, new_alpha_gam, new_beta_gam, new_lam, new_pi, new_weight_edge, new_weight_vertex,\
method, likelihood, num_of_clusters = inits[init_with_highest_ll, :]


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


LR_test(data, predicted_cluster, dlp_l)