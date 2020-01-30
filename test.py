import numpy as np
from CopyMix import util, initialization
from sklearn.metrics.cluster import v_measure_score
import csv
from numpy.linalg import norm


def calculate_predicted_c(param):
    pi, weight_vertex = param
    k = np.where(pi == max(pi))[0]
    M = len(weight_vertex[0, 0, :])
    if len(k) > 1:
        predicted_c = [np.argmax(weight_vertex[k[0], :, m]) for m in range(M)]
    else:
        predicted_c = [np.argmax(weight_vertex[k, :, m]) for m in range(M)]

    return predicted_c


def calculate_discordant_positions(param):
    predicted_c, true_c = param
    return np.mean(predicted_c != true_c)


def calculate_average_discordant_positions(pi, weight_vertex, true_c):
    N = len(pi[:, 0])
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())

    predicted_c = pool.map(calculate_predicted_c, [(pi[n], weight_vertex) for n in range(N)])
    DP = pool.map(calculate_discordant_positions, [(predicted_c[n], true_c[n]) for n in range(N)])

    pool.close()

    return np.mean(DP), predicted_c


def elbow_finder(y_values):
    x_values = np.arange(len(y_values))
    # Max values to create line
    max_x_x = np.max(x_values)
    max_x_y = y_values[max_x_x]
    max_y_y = np.max(y_values)
    max_y_x = np.where(y_values == max_y_y)[0][0]

    # Distance from point to line
    distances = []
    for i in range(len(x_values)):
        p3 = [i, y_values[i]]
        p1 = [max_y_x, max_y_y]
        p2 = [max_x_x, max_x_y]
        d = norm(np.cross(np.array(p2) - np.array(p1), np.array(p1) - np.array(p3))) / norm(np.array(p2) - np.array(p1))
        distances = np.append(distances, d)


    # Max distance point
    x_max_dist = x_values[np.where(distances == np.max(distances))[0][0]]

    return x_max_dist


# Test configuration A in Fig. 3 for one dataset (change dataset by changing s for the seed)
def test(s, num_of_cells, seq_len, trans_1, trans_2, start_1, start_2, weight_initial):

    ############ generate data ############
    np.random.seed(s)

    Z = util.generate_Z([0.5, 0.5], num_of_cells)

    rates = np.random.uniform(low=80, high=100, size=(num_of_cells,))

    index_of_cells_cluster_1 = [index for index, value in enumerate(Z) if value == 1]
    index_of_cells_cluster_2 = [index for index, value in enumerate(Z) if value == 2]

    rates_of_cluster_1 = rates[index_of_cells_cluster_1]
    rates_of_cluster_2 = rates[index_of_cells_cluster_2]

    num_of_states = len(weight_initial[0])

    C1, Y1 = util.generate_hmm(rates_of_cluster_1, num_of_states, trans_1, start_1, seq_len)
    C2, Y2 = util.generate_hmm(rates_of_cluster_2, num_of_states, trans_2, start_2, seq_len)

    new_Y1 = np.swapaxes(Y1, 0, 1)
    new_Y2 = np.swapaxes(Y2, 0, 1)
    data = np.concatenate((new_Y1, new_Y2), axis=0)
    label_0 = [0 for i in range(len(Y1[0]))]
    label_1 = [1 for j in range(len(Y2[0]))]
    labels = np.concatenate((label_0, label_1))

    c_1 = [C1 for i in range(len(Y1[0]))]
    c_2 = [C2 for i in range(len(Y2[0]))]
    true_c = np.concatenate((c_1, c_2))
    ############ end of generate data ############

    id = '45_800_' + str(s)
    num_of_clusters_DIC = 4
    initialization.generate_initializations(id, data, num_of_states, num_of_clusters_DIC)

    import pickle
    f = open('./inits_' + id, 'rb')
    data = pickle.load(f)
    f.close()

    arr = np.reshape(data, (164, 16))

    dics = []
    for k in range(1, num_of_clusters_DIC+1):
        likelihoods = arr[np.where(arr[:, -2] == k), -3]
        good_init_index = np.where(arr[:, -3] == np.min(likelihoods))[0][0]
        dics = np.append(dics, arr[good_init_index, -5])

    interesting_dic = dics[elbow_finder(dics)]

    alpha, epsilon_r, lam, pi, epsilon_s, weight_initial, weight_vertex, weight_edge, weight_vertex_new, pi_new, \
    alpha_new, dic, method, likelihood, num_of_clusters, post = arr[np.where(arr[:, -5] == interesting_dic)][0]

    print('dic: {}'.format(dic))

    print('method: {}'.format(method))

    print("final pi:")
    print(pi_new)

    if len(alpha_new) == 1:
        v_measure = v_measure_score(labels, pi_new.flatten())
    else:
        predicted_cluster = []
        for n in range(num_of_cells):
            cell = np.float64(pi_new[n])
            predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell)))
        v_measure = v_measure_score(labels, predicted_cluster)

    if len(alpha) == 1:
        v_measure_init = v_measure_score(labels, pi.flatten())
    else:
        predicted_cluster = []
        for n in range(num_of_cells):
            cell = np.float64(pi[n])
            predicted_cluster = np.append(predicted_cluster, np.where(cell == max(cell)))
        v_measure_init = v_measure_score(labels, predicted_cluster)

    dp, predicted_c = calculate_average_discordant_positions(pi_new, post, true_c)

    dp_init, predicted_c_init = calculate_average_discordant_positions(pi, weight_vertex, true_c)

    print('V_measure of the chosen initialization method: {}'.format(v_measure_init))
    print('V_measure of CopyMix: {}'.format(v_measure))
    print('Discordant positions of the chosen initialization method: {}'.format(dp_init))
    print('Discordant positions of CopyMix: {}'.format(dp))

    with open('./inits_' + id + '_result', 'w', encoding='US-ASCII') as f:
        writer = csv.writer(f)
        writer.writerow(['v_measure', 'average_discordant_positions', 'initialisation_method', 'v_measure_init',
                         'average_discordant_positions_init'])
        writer.writerow([v_measure, dp, method, v_measure_init, dp_init])


test(0, 45, 800, [[.7, .3], [.7, .3]], [[.3, .7], [.3, .7]], [.1, .9], [.9, .1],
      [[0, 1], [1, 0]])