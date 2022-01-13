import numpy as np
import csv
from sklearn.metrics.cluster import v_measure_score

dlp_labels = np.empty((0, 2), dtype=object)
with open('/Users/negar/PycharmProjects/Test/CopyMix_Gaussian/ginkgo/ov2295_clone_clusters_dlp.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        dlp_labels = np.vstack((dlp_labels, row))

dlp_l = dlp_labels[1:, 1].flatten()

ginkgo_labels = np.empty((0, 2), dtype=object)
with open('/Users/negar/PycharmProjects/Test/CopyMix_Gaussian/ginkgo/ov2295_clone_clusters_ginkgo.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        ginkgo_labels = np.vstack((ginkgo_labels, row))

ginkgo_l = ginkgo_labels[:, 1].flatten()

print(v_measure_score(dlp_l, ginkgo_l))