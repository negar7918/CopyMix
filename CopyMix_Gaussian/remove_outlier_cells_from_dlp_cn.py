import numpy as np
import csv

input = np.empty((0, 6207), dtype=object)
with open('/Users/negar/PycharmProjects/Test/CopyMix/CopyMix_Gaussian/ov2295_cn_swapped.csv', encoding='US-ASCII') as f:
    reader = csv.reader(f)
    for row in reader:
        input = np.vstack((input, row))

dlp_cn = np.array((input[:, 1:]))

dlp_labels = np.empty(1966, dtype=object)
with open('/Users/negar/Downloads/ov2295_clone_clusters.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if np.any(np.where(input == row[0])) == False:
            continue
        dlp_labels[np.where(input == row[0])[0][0]] = row[1]
dlp_labels[dlp_labels == None] = 'na'

to_delete = np.where(dlp_labels == 'na')[0]
print(to_delete)
dlp_cn = np.delete(dlp_cn, to_delete, 0)

np.savetxt("./dlp_copy_numbers_removied_outlier_cells.csv", dlp_cn.astype(int), delimiter=",")
