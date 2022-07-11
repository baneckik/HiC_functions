import hicstraw
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# ---------------------- Extracting Hi-C (from Sebastian) -----------------------

def extract_hic(path, res, chrom1, chrom2, normalization="NONE"):
    # assumes KR normalization and BP resolutions
    result = hicstraw.straw("observed", normalization, path, chrom1, chrom2, "BP", res)
    row_indices, col_indices, data = list(), list(), list()
    for record in tqdm(result):
        row_indices.append(record.binX)
        col_indices.append(record.binY)
        data.append(record.counts)
        if record.binX != record.binY:
            row_indices.append(record.binY)
            col_indices.append(record.binX)
            data.append(record.counts)
    row_indices = np.asarray(row_indices) / res
    col_indices = np.asarray(col_indices) / res
    max_size = int(max(np.max(row_indices), np.max(col_indices))) + 1
    matrix = coo_matrix((data, (row_indices.astype(int), col_indices.astype(int))),
                            shape=(max_size, max_size)).toarray()
    matrix[np.isnan(matrix)] = 0
    matrix[np.isinf(matrix)] = 0
    matrix_full = matrix.T + matrix  # make matrix symmetric
    np.fill_diagonal(matrix_full, np.diag(matrix))  # prevent doubling of diagonal from prior step
    #np.save(f'test/matrix_{self.name}.npy',matrix_full)
    return matrix_full
    
    
def extract_hic2(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["x", "y", "val"]
    
    diffs = [abs(df.x[i]-df.x[i-1]) for i in range(1, 10) if df.x[i]!=df.x[i-1]]
    diff = np.min(diffs)
    min_val = 0 #min(np.min(df.x), np.min(df.y))
    max_val = min(np.max(df.x.iloc[-20:]), np.max(df.y.iloc[-20:]))
    n = (max_val-min_val)//diff+1
    array = np.zeros((n, n))
    for i in range(df.shape[0]):
        xi = (df.iloc[i, 0]-min_val)//diff
        yi = (df.iloc[i, 1]-min_val)//diff
        if xi<n and yi<n:
            array[xi, yi] = df.iloc[i, 2]
            array[yi, xi] = df.iloc[i, 2]
    return diff, n, array

