"""Numerical result summaries extracted from PDGNNs visualize.py; no plotting dependency."""
import pickle
import numpy as np

def AP_err(performance_matrices):
    # given a list of performence matrices, return the APs the errors (range), and std
    n_tasks = performance_matrices[0].shape[0]
    performance_means_all_repeats = np.stack([[np.mean(m[i,0:i+1]) for i in range(n_tasks)] for m in performance_matrices])
    performance_means = np.mean(performance_means_all_repeats, axis=0)
    err_all = performance_means_all_repeats - performance_means
    std = performance_means_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return performance_means, err, std

def AF(acc_matrix):
    # given a acc matrix, return AF
    n_tasks = acc_matrix.shape[0]
    backward = []
    for t in range(n_tasks - 1):
        b = acc_matrix[n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(b)
    return np.mean(backward)

def AF_err(performance_matrices):
    # given a list of acc matrices, return the AMs and the errors
    AF_all_repeats = []
    for m in performance_matrices:
        AF_all_repeats.append(AF(m))
    AF_mean = np.mean(AF_all_repeats)
    AF_all_repeats = np.stack(AF_all_repeats)
    err_all = AF_all_repeats - AF_mean
    std = AF_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return AF_mean, err, std

def show_final_APAF(result_path, GCGL=False):
    """
        The function to show the final AP and AF. Output are orgnized in a LaTex firendly way.

        :param result_path: The path to the experimental result
        """
    #show the final AP and AF
    performance_matrices = pickle.load(open(result_path, 'rb'))
    performance_mean, err, std_am = AP_err(performance_matrices)

    # AF
    AF_mean, err_AF, std_AF = AF_err(performance_matrices)

    if not GCGL:
        output_str=r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF)
        print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1], std_am[-1], AF_mean, std_AF))
    else:
        output_str=r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1]*100, std_am[-1]*100, AF_mean*100, std_AF*100)
        print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(performance_mean[-1]*100, std_am[-1]*100, AF_mean*100, std_AF*100)) # convert GCGL results to percentages
    return output_str
