import numpy as np
import scanpy as sc
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def train_val_split(adata, seed, train_size=0.8, val_size=0.1, test_size=0.1):
    assert train_size + val_size + test_size == 1

    adata = adata.copy()
    np.random.seed(seed)

    cell_nums = adata.n_obs
    test_val = np.random.choice(cell_nums, int(cell_nums * (val_size + test_size)), replace=False)
    idx_train = [i for i in list(range(cell_nums)) if i not in test_val]
    idx_test = np.random.choice(test_val, int(len(test_val) * (test_size / (val_size + test_size))), replace=False)
    idx_val = [i for i in test_val if i not in idx_test]

    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_train] = True
    adata.obs['idx_train'] = tmp
    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_val] = True
    adata.obs['idx_val'] = tmp
    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_test] = True
    adata.obs['idx_test'] = tmp

    return adata