from os import listdir
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from ..utils import generate_negatives_lanl, generate_negatives
# ----------------------------------------------------------------------------
# Raw traces loading and edges extraction
# ----------------------------------------------------------------------------
def _load_single_file(x):
    df = pd.read_csv(x, header=None, 
                        names=["time", "src", "dst", "anomaly"])
    # Swap columns to have lower IP first
    idx = df['dst'] < df['src']
    df.loc[idx, ['src','dst']] = df.loc[idx, ['dst','src']].values
    df["count"] = 1

    return df

def _create_edges(df):
    edges = pd.pivot_table(df, 
                        index=["src", "dst"], 
                        aggfunc="sum")[["count", "anomaly"]]\
            .reset_index()
    edges = edges.loc[edges.src!=edges.dst]
    
    return edges

def load_edges(lanl_path):
    # Manage file names and sort them   
    flist = listdir(lanl_path)
    flist.sort(key=lambda x: float(x[:-4]) if x.endswith(".txt") else .0)
    traces = []
    for _file in tqdm(flist):
        if _file.endswith(".txt"):
            df = _load_single_file(f'{lanl_path}/{_file}')
            df = _create_edges(df)
            traces.append(df)
    
    return traces

# ----------------------------------------------------------------------------
# Sparse adjacency matrices retrieval
# ----------------------------------------------------------------------------
def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 1e-15
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def _get_sparse_matrices(edges, n_nodes):
    """Manage indices and values for sparse matrices."""
    indices = [np.stack([
                    # Source nodes
                    np.concatenate((x.src.values, # Original edges
                                    x.dst.values, # Symmetric edges
                                    np.arange(n_nodes))), # Self-loops
                    # Destination nodes
                    np.concatenate((x.dst.values, # Original edges
                                    x.src.values, # Symmetric edges
                                    np.arange(n_nodes))) # Self-loops
            ]) for x in edges]
    values = [np.concatenate((x["count"].values.reshape(-1,), 
                            x["count"].values.reshape(-1,), 
                            np.ones([n_nodes,]))) for x in edges]
    return indices, values

def get_adjacency_matrices(edges):
    # Get total number of nodes
    n_nodes = max([max(x.src.max(), x.dst.max()) for x in edges]) + 1
    # Extract sparse matrices
    indices, values = _get_sparse_matrices(edges, n_nodes)

    adjs, anomalies_idx_list = [], []
    for i, zipped in enumerate(zip(indices, values)):
        idx, vals = zipped
        # Get the sparse adjacency matrix
        adj = _to_sparse_tensor(
                _normalize(
                    sp.coo_matrix((vals, (idx[0], idx[1])), 
                            shape=(n_nodes, n_nodes),
                            dtype=np.float32)))
        adjs.append(adj)
        # Manage anomalies
        anomalies_idx = edges[i].anomaly.astype(bool).values
        new_anomalies_idx = np.concatenate((anomalies_idx,
                                            anomalies_idx,
                                            np.zeros([n_nodes,], 
                                                    dtype=bool)), axis=0)
        sorted_idx = np.lexsort((idx[1], idx[0]))
        anomalies_idx_list.append(new_anomalies_idx[sorted_idx])

    return adjs, anomalies_idx_list





def get_self_supervised_edges(X, n_days, cuda, bipartite=False, ns_edge=1):
    """
    Generate negative edges. If bipartite = True, negative edges
    are only generated between source and destination nodes.
    """
    
    n_edges = len(X[n_days+1].coalesce().values())
    idx = np.arange(n_edges)
    index = X[n_days+1].coalesce().indices()
    values = X[n_days+1].coalesce().values()
    filtered_index = index[:, index[0] < index[1]]
    
    # Sample negative edges
    if bipartite:
        active_sources = torch.unique(filtered_index[0])
        active_dest = torch.unique(filtered_index[1])
        neg_edges = generate_negatives(ns_edge * filtered_index.shape[1], active_sources.cpu().numpy(),
                                       active_dest.cpu().numpy(), index.cpu().numpy()).T
    else:
        active_nodes = torch.unique(index[0])
        if cuda:
            active_nodes = active_nodes.cpu()
        neg_edges = generate_negatives_lanl(filtered_index.shape[1], 
                                        active_nodes.cpu().numpy(), 
                                        index.cpu().numpy()).T
    
    if cuda:
        return neg_edges.cuda(), index.cuda()
    else:    
        return neg_edges, index