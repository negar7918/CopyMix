import numpy as np
import csv

import pandas as pd
import scipy.stats as stats
from scipy.stats.distributions import chi2
import pickle
from sklearn.metrics.cluster import v_measure_score
from CopyMix.CopyMix_Gaussian import util
import matplotlib.pyplot as plt
import matplotlib as mpl


def cocluster(predicted_cluster, num_of_cells, dlp_l):
    heatmap_clustering = np.zeros((3, 9))
    i = 0
    l = np.array([0, 1, 2])
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
    plt.ylabel('3 cell lines')
    plt.xlabel('9 clones')
    plt.title('Counts per cell line per dlp-method cluster')
    plt.colorbar()
    plt.savefig('/Users/negar/PycharmProjects/Test/CopyMix/CopyMix_Gaussian/plots/celllines.png')

    print(heatmap_clustering)
    print('\n')


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

#cocluster(predicted_cluster, num_of_cells, dlp_l)

celllines = dlp_l.copy()
celllines[np.logical_or(celllines=='E', celllines=='F')] = 0
celllines[np.logical_or(np.logical_or(celllines=='G', celllines=='H'), celllines=='I')] = 1
celllines[np.logical_or(np.logical_or(np.logical_or(celllines=='A', celllines=='B'), celllines=='C'), celllines=='D')] = 2
cocluster(celllines, num_of_cells, dlp_l)

#
# import scipy.spatial.distance as dst
# import scipy.cluster.hierarchy as sch
# def _secondary_clustering(data):
#     D = dst.squareform(dst.pdist(data.T, 'cityblock'))
#     Y = sch.linkage(D, method='complete')
#     Z = sch.dendrogram(Y, color_threshold=-1, no_plot=True)
#     idx = np.array(Z['leaves'])
#     ordering = np.zeros(idx.shape[0], dtype=int)
#     ordering[idx] = np.arange(idx.shape[0])
#     return


from matplotlib.colors import ListedColormap
color_reference = {0:'#3182BD', 1:'#9ECAE1', 2:'#CCCCCC', 3:'#FDCC8A', 4:'#FC8D59', 5:'#E34A33', 6:'#B30000', 7:'#980043', 8:'#DD1C77', 9:'#DF65B0', 10:'#C994C7', 11:'#D4B9DA'}
def get_cn_cmap(cn_data):
    min_cn = int(cn_data.min())
    max_cn = int(cn_data.max())
    assert min_cn - cn_data.min() == 0
    assert max_cn - cn_data.max() == 0
    color_list = []
    for cn in range(min_cn, max_cn+1):
        if cn > max(color_reference.keys()):
            cn = max(color_reference.keys())
        color_list.append(color_reference[cn])
    return ListedColormap(color_list)


# import pandas as pd
# def plot_clustered_cell_cn_matrix(ax, cn_data, cn_field_name, cluster_field_name='cluster_id', raw=False, max_cn=13):
#     plot_data = cn_data.merge(pd.Series([str(a) for a in range(1, 23)] + ['X', 'Y'])) #cn_data.merge(refgenome.info.chrom_idxs)
#     plot_data = plot_data.set_index(['chr_index', 'start', 'cell_id', cluster_field_name])[cn_field_name].unstack(level=['cell_id', cluster_field_name]).fillna(0)
#
#     ordering = _secondary_clustering(plot_data.values)
#     ordering = pd.Series(ordering, index=plot_data.columns, name='cell_order')
#     plot_data = plot_data.T.set_index(ordering, append=True).T
#
#     plot_data = plot_data.sort_index(axis=1, level=[1, 2])
#     if max_cn is not None:
#         plot_data[plot_data > max_cn] = max_cn
#
#     mat_chrom_idxs = plot_data.index.get_level_values(0).values
#     chrom_boundaries = np.array([0] + list(np.where(mat_chrom_idxs[1:] != mat_chrom_idxs[:-1])[0]) + [plot_data.shape[0] - 1])
#     chrom_sizes = chrom_boundaries[1:] - chrom_boundaries[:-1]
#     chrom_mids = chrom_boundaries[:-1] + chrom_sizes / 2
#     ordered_mat_chrom_idxs = mat_chrom_idxs[np.where(np.array([1] + list(np.diff(mat_chrom_idxs))) != 0)]
#     chrom_names = np.array([str(a) for a in range(1, 23)] + ['X', 'Y'])[ordered_mat_chrom_idxs]
#
#     mat_cluster_ids = plot_data.columns.get_level_values(1).values
#     cluster_boundaries = np.array([0] + list(np.where(mat_cluster_ids[1:] != mat_cluster_ids[:-1])[0]) + [plot_data.shape[1] - 1])
#     cluster_sizes = cluster_boundaries[1:] - cluster_boundaries[:-1]
#     cluster_mids = cluster_boundaries[:-1] + cluster_sizes / 2
#
#     cmap = None
#     if not raw:
#         cmap = get_cn_cmap(plot_data.values)
#
#     im = ax.imshow(plot_data.astype(float).T, aspect='auto', cmap=cmap, interpolation='none')
#
#     ax.set(xticks=chrom_mids)
#     ax.set(xticklabels=chrom_names)
#
#     for val in chrom_boundaries[:-1]:
#         ax.axvline(x=val, linewidth=1, color='black', zorder=100)
#
#     return plot_data
#
#
# def get_cluster_palette(n_col):
#     if n_col <= 10:
#         palette = plt.get_cmap("tab10")
#     elif n_col <= 21:
#         palette = mpl.colors.ListedColormap([
#             '#1d1d1d', '#ebce2b', '#702c8c', '#db6917', '#96cde6', '#ba1c30',
#             '#c0bd7f', '#7f7e80', '#5fa641', '#d485b2', '#4277b6', '#df8461',
#             '#463397', '#e1a11a', '#91218c', '#e8e948', '#7e1510', '#92ae31',
#             '#6f340d', '#d32b1e', '#2b3514'
#         ])
#     else:
#         palette = plt.get_cmap("hsv")
#     return
#
# import itertools
# def get_cluster_color_map(cluster_ids):
#     num_colors = len(np.unique(cluster_ids[cluster_ids >= 0]))
#     pal = get_cluster_palette(num_colors)
#
#     color_map = {}
#
#     cluster_ids = np.sort(np.unique(cluster_ids))
#     for cluster_id in np.sort(np.unique(cluster_ids)):
#         if cluster_id < 0:
#             color_map[cluster_id] = (0.75, 0.75, 0.75, 1.0)
#
#     cluster_ids = cluster_ids[cluster_ids >= 0]
#
#     idx = 0.
#     for cluster_id in itertools.chain(cluster_ids[::2], cluster_ids[1::2]):
#         color_map[cluster_id] = pal(float(idx) / float(num_colors - 1))
#         idx += 1
#
#     return color_map
#
#
# def get_cluster_colors(cluster_ids):
#     color_map = get_cluster_color_map(cluster_ids)
#
#     color_mat = []
#     for cluster_id in cluster_ids:
#         color_mat.append(color_map[cluster_id])
#
#     return
#
#
# def plot_clustered_cell_cn_matrix_figure(fig, cn_data, cn_field_name, cluster_field_name='cluster_id', raw=False, max_cn=13):
#     ax = fig.add_axes([0.1,0.0,0.9,1.])
#     plot_data = plot_clustered_cell_cn_matrix(ax, cn_data, cn_field_name, cluster_field_name=cluster_field_name, raw=raw, max_cn=max_cn)
#
#     cluster_ids = plot_data.columns.get_level_values(1).values
#     color_mat = get_cluster_colors(cluster_ids)
#
#     ax = fig.add_axes([0.0,0.0,0.05,1.])
#     ax.imshow(np.array(color_mat)[::-1, np.newaxis], aspect='auto', origin='lower', interpolation='none')
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     return


c = util.calculate_most_probable_states(data, trans, new_weight_vertex, weight_initial, new_pi)
#c = c.swapaxes(1, 0)
new_c = np.empty((4, 6206))
new_c[0] = c[0]
new_c[1] = c[2]
new_c[2] = c[5]
new_c[3] = c[6]

new_c = new_c.swapaxes(1, 0)

cluster_col = 'cluster_id'
#fig = plt.figure(figsize=(10, 15))
#matrix_data = plot_clustered_cell_cn_matrix_figure(fig, plot_data, 'state', cluster_field_name=cluster_col)

cmap = get_cn_cmap(new_c)
fig, ax = plt.subplots()
im = ax.imshow(new_c.astype(float).T, aspect='auto', cmap=cmap, interpolation='none')
#plt.show()
ax.set_xlabel('bins')
ax.set_ylabel('clones')
ax.set_yticks([0, 1, 2, 3])
plt.savefig('/Users/negar/PycharmProjects/Test/CopyMix/CopyMix_Gaussian/plots/CopyMix_estimated_copy_number_for_dlp.png')

#for val in chrom_boundaries[:-1]:
#    ax.axvline(x=val, linewidth=1, color='black', zorder=100)

copymix_cells_c = np.empty((891, 6206))

indexes = np.where(new_pi[:, 0] > (1 / 4))[0]
copymix_cells_c[indexes] = np.ones((len(indexes), 6206)) * c[0]
print((len(indexes)))

indexes_1 = np.where(new_pi[:, 2] > (1 / 4))[0]
copymix_cells_c[indexes_1] = np.ones((len(indexes_1), 6206)) * c[2]
print((len(indexes_1)))

indexes_2 = np.where(new_pi[:, 5] > (1 / 4))[0]
copymix_cells_c[indexes_2] = np.ones((len(indexes_2), 6206)) * c[5]
print((len(indexes_2)))

indexes_3 = np.where(new_pi[:, 6] >= (1 / 4))[0]
copymix_cells_c[indexes_3] = np.ones((len(indexes_3), 6206)) * c[6]
print((len(indexes_3)))

dlp_cn = np.empty((0, 6206), dtype=object)
with open('./dlp_copy_numbers_removied_outlier_cells.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        dlp_cn = np.vstack((dlp_cn, row))

tv = util.calculate_total_variation(copymix_cells_c[1:, :], dlp_cn.astype('float'))
print('TV distance between copy numbers:' + str(tv))

new_dlp_cn = dlp_cn.astype('float').swapaxes(1, 0)
cmap = get_cn_cmap(new_dlp_cn)
fig, ax = plt.subplots()
im = ax.imshow(new_dlp_cn.astype(float).T, aspect='auto', cmap=cmap, interpolation='none')
#plt.show()
ax.set_xlabel('bins')
ax.set_ylabel('cells')
plt.savefig('/Users/negar/PycharmProjects/Test/CopyMix/CopyMix_Gaussian/plots/plot_dlp_copy_numbers.png')

copymix_cells_c = copymix_cells_c.swapaxes(1, 0)
cmap = get_cn_cmap(copymix_cells_c)
fig, ax = plt.subplots()
im = ax.imshow(copymix_cells_c.astype(float).T, aspect='auto', cmap=cmap, interpolation='none')
#plt.show()
ax.set_xlabel('bins')
ax.set_ylabel('cells')
plt.savefig('/Users/negar/PycharmProjects/Test/CopyMix/CopyMix_Gaussian/plots/plot_copymix_cells_copy_numbers.png')


