import numpy as np

def truncate(cov_path, q_min, q_max):
    old_cov = np.loadtxt(cov_path, unpack=True)

    # removing extra bins
    mask_i = (q_min<=old_cov[0]) & (old_cov[0]<=q_max)
    mask_j = (q_min<=old_cov[1]) & (old_cov[1]<=q_max)
    mask = mask_i & mask_j
    cov = old_cov[:, mask]

    # removing hexadecapole
    n_q = int(np.sqrt(len(cov[2]))/3)
    cov_matrix_complete = cov[2]
    cov_matrix_complete = np.reshape(cov_matrix_complete, (3*n_q, 3*n_q))
    cov_matrix = cov_matrix_complete[:2*n_q, :2*n_q]

    return cov_matrix
