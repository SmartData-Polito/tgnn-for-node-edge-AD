import torch
import numpy as np
import random
from tqdm import tqdm
import scipy.sparse as sp

def compute_accuracy(y_pred, y_true):
    y_pred = torch.round(y_pred)

    return torch.sum(y_pred == y_true) / len(y_true)

def get_set_diff(A,B):
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    return np.array([x for x in aset.difference(bset)])

def generate_negatives(anomaly_num, active_source, active_dest, real_edges):
    """
    p: percentage of anomalies wrt existing edges
    """
    n_extractions = 10 * anomaly_num
    idx_1 = np.expand_dims(np.transpose(np.random.choice(active_source, n_extractions)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(active_dest, n_extractions)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1).astype(real_edges.dtype)
    # remove duplicates and existing edges
    fake_edges = np.unique(fake_edges, axis=0) 
    fake_edges = get_set_diff(fake_edges, real_edges.T)
    neg_edges = fake_edges[:anomaly_num, :]

    return torch.tensor(neg_edges)


def generate_negatives_lanl(anomaly_num, active_nodes, real_edges):
    """
    p: percentage of anomalies wrt existing edges
    """

    n_extractions = 20 * anomaly_num
    
    idx_1 = np.expand_dims(np.transpose(np.random.choice(active_nodes, n_extractions)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(active_nodes, n_extractions)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1).astype(real_edges.dtype)
    # remove duplicates and existing edges
    fake_edges = np.unique(fake_edges, axis=0) 
    fake_edges = fake_edges[fake_edges[:,0] != fake_edges[:,1]] # remove self-loops
    fake_edges = get_set_diff(fake_edges, real_edges.T)
    neg_edges = fake_edges[:anomaly_num, :]
    
    return torch.tensor(neg_edges)

def split_edges_lanl(adjs, n_days, cuda):
    ## Divide links into 2 groups:
    # training message 50%, training supervision 50%,
    n_edges = len(adjs[n_days+1].coalesce().values())
    idx = np.arange(n_edges)
    random.Random(42).shuffle(idx)
    idx_train_msg = torch.tensor(idx[:int(0.5*n_edges)])
    idx_train_sup = torch.tensor(idx[int(0.5*n_edges):])
    index = adjs[n_days+1].coalesce().indices()
    values = adjs[n_days+1].coalesce().values()
    # Only keep edges in one direction (i.e. smaller ID -> larger ID)
    idx_train_sup = idx_train_sup[index[0][idx_train_sup] < index[1][idx_train_sup]]
    
    active_nodes = torch.unique(index[0])
    if cuda:
        active_nodes = active_nodes.cpu()
        
    # Sample negative edges
    neg_edges = generate_negatives_lanl(len(idx_train_sup), active_nodes.numpy(), index.numpy()).T
    
    ## Create adjacency matrixes for training
    A_train = torch.sparse.FloatTensor(index[:,idx_train_msg], values[idx_train_msg], adjs[n_days+1].shape)
    
    if cuda:
        return idx_train_msg.cuda(), idx_train_sup.cuda(), neg_edges.cuda(), A_train.cuda(), index.cuda()
    else:    
        return idx_train_msg, idx_train_sup, neg_edges, A_train, index

    

def split_edges(adjs, n_days, cuda, n_ips):
    ## Divide links into 2 groups:
    # training message 50%, training supervision 50%,
    n_edges = len(adjs[n_days+1].coalesce().values())
    idx = np.arange(n_edges)
    random.Random(42).shuffle(idx)
    idx_train_msg = torch.tensor(idx[:int(0.5*n_edges)])
    idx_train_sup = torch.tensor(idx[int(0.5*n_edges):])
    index = adjs[n_days+1].coalesce().indices()
    values = adjs[n_days+1].coalesce().values()
    # Only keep IP-to-port edges
    idx_train_sup = idx_train_sup[index[0][idx_train_sup] < index[1][idx_train_sup]]
    active_ips = torch.unique(index[0][index[0] < n_ips])
    active_ports = torch.unique(index[0][index[0] >= n_ips])
    if cuda:
        active_ips, active_ports = active_ips.cpu(), active_ports.cpu()
        index, values = index.cpu(), values.cpu()
    
    # Sample negative edges
    neg_edges = generate_negatives(len(idx_train_sup), active_ips.numpy(), active_ports.numpy(), index.numpy()).T
    
    ## Create adjacency matrixes for training
    A_train = torch.sparse.FloatTensor(index[:,idx_train_msg], values[idx_train_msg], adjs[n_days+1].shape)
    
    if cuda:
        return idx_train_msg.cuda(), idx_train_sup.cuda(), neg_edges.cuda(), A_train.cuda(), index.cuda()
    else:    
        return idx_train_msg, idx_train_sup, neg_edges, A_train, index

    
def symmetric_sparse_matrix(mx):
    index = mx.coalesce().indices()
    values = mx.coalesce().values()
    n_edges = index.shape[1]
    
    new_index = torch.zeros([2,2*n_edges], dtype=torch.long)
    new_index[:,:n_edges] = index
    new_index[:,n_edges:] = torch.flip(index, [0])
    new_values = torch.cat([values, values], dim=0)
    
    return torch.sparse.FloatTensor(new_index, new_values, mx.shape)
    

def perturb_adjacency(adjs, start_day, end_day, bipartite=False, cuda=False):
    pert_adjs = []
    n_nodes = adjs[0].shape[0]
    for day in range(start_day, end_day + 1):
        index = adjs[day].coalesce().indices()
        # Naive solution: select ALL destination nodes at random
        if bipartite: # Start from directed edges user -> item
            pert_index = index[:, index[0] < index[1]]
            active_dest = torch.unique(index[1, index[0] < index[1]])
            pert_index[1] = torch.tensor(np.random.choice(active_dest.cpu().numpy(), pert_index.shape[1]))
        else: 
            pert_index = torch.zeros([2, index.shape[1] - n_nodes]) 
            pert_index[0] = index[0, index[0] != index[1]]
            active_nodes = torch.unique(index[0])
            # Using .cpu() here doesn't change anything if tensor in not on gpu
            pert_index[1] = torch.tensor(np.random.choice(active_nodes.cpu().numpy(), pert_index.shape[1]))
        
        if cuda:
            pert_adjs.append(torch.sparse.FloatTensor(pert_index.long().cuda(), torch.ones([pert_index.shape[1],]).cuda(), (n_nodes, n_nodes)).cuda())
        else:
            pert_adjs.append(torch.sparse.FloatTensor(pert_index.long(), torch.ones([pert_index.shape[1],]), (n_nodes, n_nodes)))
    return pert_adjs
    
    
def generate_val_test_lab_nodes(adjs, train_days, eval_days, final_day, anomalies_edges_idx, anomalies_thr=1, anomalies_nodes_idx=None,bipartite=False):
    """
    Generate label vectors for validation and test set for nodes.
    PARAMS:
    anomalies_thr: Minimum number of anomalous connections in the same snapshot
                    for a node to be considered anomalous in that snapshot
    """
    
    val_lab = []
    test_lab = []
    n_nodes = adjs[0].shape[0]
    
    for d in range(train_days+1, train_days + eval_days+1):
        if anomalies_nodes_idx is not None:
            index = adjs[d].coalesce().indices()
            active_nodes = index[0, index[0] != index[1]].unique()
            val_lab.append(torch.tensor(~anomalies_nodes_idx[d])[active_nodes])
        else:
            index = adjs[d].coalesce().indices()
            active_nodes = index[0, index[0] != index[1]].unique()
            lab = torch.zeros([n_nodes,], dtype=torch.long)
            # Set both source and destination of anomalous edges
            # as anomalous nodes
            an_conn = torch.bincount(index[0, torch.tensor(anomalies_edges_idx[d])], minlength=n_nodes)
            lab = (an_conn < anomalies_thr).long()
            val_lab.append(lab[active_nodes])
        
    for d in range(train_days + eval_days+1, final_day+1):
        if anomalies_nodes_idx is not None:
            index = adjs[d].coalesce().indices()
            active_nodes = index[0, index[0] != index[1]].unique()
            test_lab.append(torch.tensor(~anomalies_nodes_idx[d])[active_nodes])
        else:
            index = adjs[d].coalesce().indices()
            active_nodes = index[0, index[0] != index[1]].unique()
            lab = torch.zeros([n_nodes,], dtype=torch.long)
            # Set both source and destination of anomalous edges
            # as anomalous nodes
            an_conn = torch.bincount(index[0, torch.tensor(anomalies_edges_idx[d])], minlength=n_nodes)
            lab = (an_conn < anomalies_thr).long()
            test_lab.append(lab[active_nodes])
    
    return torch.cat(val_lab), torch.cat(test_lab)
    
def generate_val_test_lab_edges(X, train_days, eval_days, final_day, anomalies_idx):
    val_lab = []
    test_lab = []
    for d in range(train_days+1, train_days + eval_days+1):
        index = X[d].coalesce().indices()
        y_val = torch.ones([(index[0] < index[1]).sum(),1])
        y_val[anomalies_idx[d][index[0] < index[1]]] = 0
        val_lab.append(y_val)
    for d in range(train_days + eval_days+1, final_day+1):
        index = X[d].coalesce().indices()
        y_test = torch.ones([(index[0] < index[1]).sum(),1])
        y_test[anomalies_idx[d][index[0] < index[1]]] = 0
        test_lab.append(y_test)
    
    return torch.cat(val_lab), torch.cat(test_lab)


def compute_fpr_tpr(pred, lab, th=.5):
    pred_lab = torch.where(torch.exp(pred) > th, 1, 0)
    tpr = (pred_lab[lab == 0] == 0).sum() / (lab == 0).sum()
    fpr = (pred_lab[lab == 1] == 0).sum() / (lab == 1).sum()
    return fpr.item(), tpr.item()

def get_set_diff(A,B):
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    return np.array([x for x in aset.difference(bset)])

def generate_anomalies_edges(p, real_edges, bipartite=True):
    """
    p: percentage of anomalies wrt existing edges
    """
    anomaly_num = int(np.floor(p * real_edges.shape[0]))
    n_extractions = 10 * anomaly_num
 
    # Sample anomalous edges
    if bipartite: # sources and destinations are separate
        active_sources = np.unique(real_edges[:,0])
        active_dest = np.unique(real_edges[:,1])
        idx_1 = np.expand_dims(np.transpose(np.random.choice(active_sources, n_extractions, replace=True)) , axis=1)
        idx_2 = np.expand_dims(np.transpose(np.random.choice(active_dest, n_extractions, replace=True)) , axis=1)
        fake_edges = np.concatenate((idx_1, idx_2), axis=1).astype(real_edges.dtype)
        # remove duplicates, self-loops and existing edges
        fake_edges = np.unique(fake_edges, axis=0) 
        fake_edges = fake_edges[fake_edges[:,0] != fake_edges[:,1]]
        fake_edges = get_set_diff(fake_edges, real_edges[:,:2])
    else: # each node can be both source or target
        active_nodes = np.unique(np.concatenate([np.unique(real_edges[:,0]), np.unique(real_edges[:,1])]))
        idx_1 = np.expand_dims(np.transpose(np.random.choice(active_nodes, n_extractions, replace=True)) , axis=1)
        idx_2 = np.expand_dims(np.transpose(np.random.choice(active_nodes, n_extractions, replace=True)) , axis=1)
        fake_edges = np.concatenate((idx_1, idx_2), axis=1).astype(real_edges.dtype)
        # order all edges as smaller ID -> larger ID, remove duplicates, self-loops and existing edges (in both directions!)
        fake_edges[fake_edges[:,0] > fake_edges[:,1]] = fake_edges[fake_edges[:,0] > fake_edges[:,1]][:,::-1]
        fake_edges = np.unique(fake_edges, axis=0) 
        fake_edges = fake_edges[fake_edges[:,0] != fake_edges[:,1]]
        fake_edges = get_set_diff(fake_edges, real_edges[:,:2])
        fake_edges = get_set_diff(fake_edges, real_edges[:,:2][:,::-1])
        
    anomalies = fake_edges[:anomaly_num, :]
    # add generated anomalies to the edge list
    idx_test = np.zeros([real_edges.shape[0] + anomaly_num, ], dtype=np.int32)
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 1
    synthetic_test = np.zeros([np.size(idx_test, 0), 2], dtype=np.int32)
    synthetic_test[idx_test == 1, :] = anomalies
    synthetic_test[idx_test == 0, :] = real_edges[:,:2]
    edge_values = 50 * np.ones([synthetic_test.shape[0],1], dtype=np.int32) # Check if this is the best choice
    edge_values[idx_test == 0] = real_edges[:,2].reshape(-1,1) 
    synthetic_test = np.concatenate((synthetic_test,edge_values), axis=1)
    return synthetic_test, idx_test 


def add_anomalies_edges(real_edges, to_add):
    """
    Add anomalous edges to the adjacency matrix.

    PARAMS:
    to_add: list of edges to add as anomalies
    """
        
    # Remove anomalous edges if they already exist
    to_add = get_set_diff(to_add, real_edges[:,:2])
    anomalies = to_add 
    anomaly_num = anomalies.shape[0]

    # add generated anomalies to the edge list
    idx_test = np.zeros([real_edges.shape[0] + anomaly_num, ], dtype=np.int32)
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 1
    synthetic_test = np.zeros([np.size(idx_test, 0), 2], dtype=np.int32)
    synthetic_test[idx_test == 1, :] = anomalies
    synthetic_test[idx_test == 0, :] = real_edges[:,:2]
    edge_values = np.ones([synthetic_test.shape[0],1], dtype=np.int32) # Check if this is the best choice
    edge_values[idx_test == 0] = real_edges[:,2].reshape(-1,1) 
    synthetic_test = np.concatenate((synthetic_test,edge_values), axis=1)
    return synthetic_test, idx_test 





def generate_anomalies_nodes(p, n_sources, n_dest, n_nodes, real_edges, prev_anomalies=None, 
                             active_percentage=.1, bipartite=True, anomalies_thr=1, fix_anomalous_edges=False):
    """
    p: percentage of nodes that become anomalies 
    """
    if bipartite:
        deg = np.bincount(real_edges[:,0], minlength=n_sources)
        active_nodes = np.unique(real_edges[:,0])
        n_active_nodes = active_nodes.shape[0]
        inactive_nodes = np.array([n for n in np.arange(n_sources) if n not in active_nodes])
    else:
        deg = np.bincount(real_edges[:,0], minlength=n_nodes) + np.bincount(real_edges[:,1], minlength=n_nodes)
        active_nodes = np.unique(np.concatenate([np.unique(real_edges[:,0]), np.unique(real_edges[:,1])]))
        n_active_nodes = active_nodes.shape[0]
        inactive_nodes = np.array([n for n in np.arange(n_sources) if n not in active_nodes])
    
    # Extract IDs of nodes that become anomalous 
    anomaly_num = int(np.floor(p * n_active_nodes))
    anomaly_num_active = int(np.floor(active_percentage * anomaly_num))
    anomaly_num_inactive = anomaly_num - anomaly_num_active
    # Extract active sources to transform into anomalies
    idx = np.arange(n_active_nodes)
    random.shuffle(idx)
    node_anomaly_idx = active_nodes[idx[:anomaly_num_active]]
    # Extract inactive sources to transform into anomalies
    idx = np.arange(inactive_nodes.shape[0])
    random.shuffle(idx)
    node_anomaly_idx = np.concatenate((node_anomaly_idx, inactive_nodes[idx[:anomaly_num_inactive]])).astype(np.int32) 
    print(f"Injecting {node_anomaly_idx.shape} anomalies")
    fake_edges = []

    # Compute how many edges to add for each node and generate them
    for an_node in node_anomaly_idx:
        if deg[an_node] < 2 * (anomalies_thr + 1) or fix_anomalous_edges:
            edges_to_add = anomalies_thr+1
        else:
            edges_to_add = np.random.randint(deg[an_node] // 2, deg[an_node] + 1)
        if bipartite:
            all_dest = np.array(list(set(np.arange(n_dest) + n_sources).difference(set(real_edges[real_edges[:,0] == an_node][:,1]))))
        else:
            all_dest = set(np.arange(n_nodes)).difference(set(real_edges[real_edges[:,0] == an_node][:,1]))
            all_dest = all_dest.difference(set(real_edges[real_edges[:,1] == an_node][:,0]))
            all_dest = np.array(list(all_dest.difference(set(node_anomaly_idx)))) # anomalies cannot be connected to other anomalies
            all_dest = all_dest[all_dest != an_node] # no self-loops
        random.shuffle(all_dest)
        dest = all_dest[:edges_to_add]
        fake_edges.append(np.stack((an_node * np.ones([len(dest),]), dest), axis=1))

    fake_edges = np.concatenate(fake_edges)

    # add generated anomalies to the edge list
    idx_test = np.zeros([real_edges.shape[0] + fake_edges.shape[0], ], dtype=np.int32)
    anomaly_pos = np.random.choice(np.size(idx_test, 0), fake_edges.shape[0], replace=False)
    idx_test[anomaly_pos] = 1
    edge_anomaly_idx = np.zeros([real_edges.shape[0] + fake_edges.shape[0], ], dtype=np.int32)
    edge_anomaly_idx[idx_test == 0] = prev_anomalies
    edge_anomaly_idx[idx_test == 1] = 1
    synthetic_test = np.zeros([np.size(idx_test, 0), 2], dtype=np.int32)
    synthetic_test[idx_test == 1, 0:2] = fake_edges
    synthetic_test[idx_test == 0, 0:2] = real_edges[:,:2]
    edge_values = np.ones([synthetic_test.shape[0],1], dtype=np.int32)
    edge_values[idx_test == 0] = real_edges[:,2].reshape(-1,1) 
    synthetic_test = np.concatenate((synthetic_test,edge_values), axis=1)
    return synthetic_test, edge_anomaly_idx, node_anomaly_idx    


def generate_anomalies_edges_temporal(p, edges, bipartite, n_intervals, train_intervals, val_intervals):
    """
    Add some random anomalous edges.
    Keep the anomalous edges that are active for a number
    of consecutive snapshots.
    
    PARAMS:
    to_keep: list of edges with the anomalies to keep
        from previous snapshot
    p: percentage of anomalies wrt existing edges
    """
    activity_len = 5
    edge_list = []
    anomalies_idx = []
    
    for interval in tqdm(range(n_intervals)):
        if interval < train_intervals or interval >= val_intervals + activity_len:
            # No anomalies to be added
            edges_with_an = edges.loc[edges.interval == interval][["user", "item", "weight"]].to_numpy()
            edge_anomaly_idx = np.zeros([edges_with_an.shape[0],], dtype=np.int32)
            node_anomaly_idx = np.array([], dtype=np.int32)
        elif interval >= train_intervals and interval < val_intervals:
            # Inject random anomalies in the validation set
            edges_with_an, edge_anomaly_idx = generate_anomalies_edges(p, 
                                        edges.loc[edges.interval == train_intervals][["user", "item", "weight"]].to_numpy(), 
                                        bipartite=bipartite)
            node_anomaly_idx = np.array([], dtype=np.int32)
        elif interval == val_intervals:
            # Define some anomalous edges to keep for some snapshots in the test set
            edges_with_an, edge_anomaly_idx = generate_anomalies_edges(p, 
                                        edges.loc[edges.interval == train_intervals][["user", "item", "weight"]].to_numpy(), 
                                        bipartite=bipartite)
            node_anomaly_idx = np.array([], dtype=np.int32)
            edges_to_keep = edges_with_an[edge_anomaly_idx.astype(bool), :2]
            
        else:
            # Keep anomalous edges
            edges_with_an, edge_anomaly_idx = add_anomalies_edges(edges.loc[edges.interval == interval][["user", "item", "weight"]].to_numpy(), edges_to_keep)
            node_anomaly_idx = np.array([], dtype=np.int32)

        edge_list.append(edges_with_an)
        anomalies_idx.append((edge_anomaly_idx, node_anomaly_idx))
    
    return edge_list, anomalies_idx
        
        
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 1e-15
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

