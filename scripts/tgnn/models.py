from torch import nn
import torch
import math
import torch.nn.functional as F
from tqdm import tqdm

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        """
               :param in_features:     size of the input per node
               :param out_features:    size of the output per node
               :param bias:            whether to add a learnable bias before the activation
               :param device:          device used for computation
        """
        # bias=False
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.sparse.mm(adj, support) 
        if self.bias is not None:
            return output + self.bias
        else:
            return output

        
class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, nfeat, nout, nhid, dropout, n_layers):
        
        super(GCN, self).__init__()
        
        self.nlayers = n_layers
        layer_input_sizes = [nfeat] + [nhid] * (self.nlayers - 1)
        layer_output_sizes = [nhid] * (self.nlayers - 1) + [nout]
        self.layers = nn.ModuleList([GraphConvolution(layer_input_sizes[i], layer_output_sizes[i]) for i in range(self.nlayers)])
        self.dropout = dropout

    def forward(self, feat, adj, cur_idx=None, verbose=False):
        x = feat
        for i in range(self.nlayers):
            x = self.layers[i](x, adj)
            if i < self.nlayers - 1:
                x = F.relu(x) 
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    

class GCNGRU(nn.Module):    
    def __init__(self, nout, nout_gru, nhid_gcn, nhid_edges, dropout, n_layers, n_nodes, cuda):
        super(GCNGRU, self).__init__()
        self.nhid = nhid_gcn
        self.n_nodes = n_nodes
        self.is_cuda = cuda
        self.nout = nout
        self.nout_gru = nout_gru
        
        self.gcn = GCN(nfeat=n_nodes, 
            nout=nout, 
            nhid=nhid_gcn, 
            dropout=dropout, 
            n_layers=n_layers)
            
        self.gru = nn.GRU(input_size=nout, hidden_size=nout_gru, num_layers=1)
        self.edge_predictor = nn.Sequential(
                                nn.Linear(2 * nout_gru, nhid_edges),
                                nn.ReLU(),
                                nn.Linear(nhid_edges, 2),
                                nn.LogSoftmax(dim=1)
                            )        
        if cuda:
            self.bn = nn.BatchNorm1d(self.nout_gru).cuda()
        else:
            self.bn = nn.BatchNorm1d(self.nout_gru)
            
        self.init_weights(self.gru)  
        self.init_weights(self.edge_predictor)
    
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, start_day, end_day, adjs, edges, ret_emb=False):
        """
        PARAMS: 
        
        start_day: initial snapshot of history to consider
        end_day: last snapshot of history (i.e. final embeddings will be at snapshot end_day + 1)
        adjs: list of adjacency matrixes (one per snapshot)
        edges: edge index of edges to predict likelihood for
        ret_emb: if True, return final and historical node embeddings; if False, return edge predictions
        """
        out = torch.zeros([end_day-start_day+1,self.n_nodes,self.nout])
        # Set feature matrix to identity 
        diag_feat = torch.sparse.FloatTensor(
                            torch.stack([torch.arange(0,self.n_nodes),
                            torch.arange(0,self.n_nodes)],dim=0), 
                            torch.ones([self.n_nodes, ]), (self.n_nodes, self.n_nodes))        
        if self.is_cuda:
            out = out.cuda()
            diag_feat = diag_feat.cuda()
        
        # Apply GCN on each daily snapshot
        for i in range(end_day-start_day+1):
            out[i] = self.gcn(diag_feat, adjs[start_day + i].float())                

        # Forward GCN outputs through GRU
        self.gru.flatten_parameters()
        _, final_emb = self.gru(out)
        final_emb = final_emb.squeeze()

        final_emb = self.bn(final_emb)
        
        # Predict edge likelihood
        nodes_first = final_emb[edges[0,:].long(), :]
        nodes_second = final_emb[edges[1,:].long(), :]
        pred = self.edge_predictor(torch.cat([nodes_first, nodes_second], dim=1))

        if ret_emb:
            return pred, out, final_emb
        
        return pred

    
    
class GCNGRU_nodes(nn.Module):    
    def __init__(self, nout, nout_gru, nhid_gcn, nhid_nodes, dropout, n_layers, n_nodes, cuda):
        super(GCNGRU_nodes, self).__init__()
        self.nhid = nhid_gcn
        self.nout = nout
        self.n_nodes = n_nodes
        self.is_cuda = cuda
        self.nout_gru = nout_gru
        
        self.gcn = GCN(nfeat=n_nodes, 
            nout=nout, 
            nhid=nhid_gcn, 
            dropout=dropout, 
            n_layers=n_layers)
            
        self.gru = nn.GRU(input_size=nout, hidden_size=nout_gru, num_layers=1)
        
        self.node_predictor = nn.Sequential(
                nn.Linear(2 * nout_gru, nhid_nodes),
                nn.ReLU(),
                nn.Linear(nhid_nodes, 2),
                nn.LogSoftmax(dim=1)
            )
            
        self.attention = nn.Sequential(
                nn.Linear(2 * nout_gru, 1),
                nn.Sigmoid()
            )
        
        if cuda:
            self.bn = nn.BatchNorm1d(self.nout_gru).cuda()
        else:
            self.bn = nn.BatchNorm1d(self.nout_gru)
        
        
    def get_node_an_scores(self, full_adj, final_emb):
        """
        Get anomaly scores for nodes given their embeddings
        and the adjacency matrix in the last snapshot
        """
        # Remove diagonal from adjacency matrix (i.e. self-loops)
        idx = full_adj.coalesce().indices()
        val = full_adj.coalesce().values()
        filtered_idx = idx[:,idx[0] != idx[1]]
        deg = torch.bincount(filtered_idx[0], minlength=full_adj.shape[0])
        diag = torch.where(deg != 0, 1/deg, 1)
        
        # Learn attention weights for neighbors
        source_nodes = final_emb[filtered_idx[0]]
        neigh_embs = final_emb[filtered_idx[1]]
        weights = self.attention(torch.cat((source_nodes, neigh_embs), dim=1))

        # Create adjacency matrix where the edge weights are the attention weights
        # normalized by the node degree
        adj = torch.sparse.FloatTensor(
            filtered_idx,
            diag[filtered_idx[0]] * weights.squeeze(),
            full_adj.shape
        )

        # Predict node anomaly score
        neighborhood_emb = torch.sparse.mm(adj, final_emb)
        pred = self.node_predictor(torch.cat([final_emb, neighborhood_emb], dim=1))
        return pred, weights

    
    def forward(self, start_day, end_day, adjs, ret_emb=False, pert_adj=None):
        """
        PARAMS: 
        
        start_day: initial snapshot of history to consider
        end_day: last snapshot of history (i.e. final embeddings will be at snapshot end_day + 1)
        adjs: list of adjacency matrixes (one per snapshot)
        ret_emb: if True, return node predictions, final and historical node embeddings and attention weights 
                if False, return node predictions
        """

        out = torch.zeros([end_day-start_day+1,self.n_nodes,self.nout])
        # Set feature matrix to identity
        diag_feat = torch.sparse.FloatTensor(
                            torch.stack([torch.arange(0,self.n_nodes),
                            torch.arange(0,self.n_nodes)],dim=0), 
                            torch.ones([self.n_nodes, ]), (self.n_nodes, self.n_nodes))
        
        if self.is_cuda:
            out = out.cuda()
            diag_feat = diag_feat.cuda()
        
        # Apply GCN on each snapshot separately
        for i in range(end_day-start_day+1):
            out[i] = self.gcn(diag_feat, adjs[start_day + i].float())
    
        # Forward node embeddings through the GRU
        self.gru.flatten_parameters()
        _, final_emb = self.gru(out)
        final_emb = final_emb.squeeze()

        final_emb = self.bn(final_emb)
        
        pred, weights = self.get_node_an_scores(adjs[end_day+1], final_emb)
        if pert_adj is not None:
            pred1, weights1 = self.get_node_an_scores(pert_adj, final_emb)
            pred = torch.cat([pred, pred1])
            weights = torch.cat([weights, weights1])
                
        if ret_emb:
            return pred, out, final_emb, weights
        
        return pred
    

class GCNGRU_both(nn.Module):    
    def __init__(self, nout, nout_gru, nhid_gcn, nhid_nodes, nhid_edges, dropout, n_layers, n_nodes, cuda):
        super(GCNGRU_both, self).__init__()
        self.nhid = nhid_gcn
        self.nout = nout
        self.n_nodes = n_nodes
        self.is_cuda = cuda
        self.nout_gru = nout_gru
        
        self.gcn = GCN(nfeat=n_nodes, 
            nout=nout, 
            nhid=nhid_gcn, 
            dropout=dropout, 
            n_layers=n_layers)
            
        self.gru = nn.GRU(input_size=nout, hidden_size=nout_gru, num_layers=1)
                
        self.node_predictor = nn.Sequential(
                nn.Linear(2 * nout_gru, nhid_nodes),
                nn.ReLU(),
                nn.Linear(nhid_nodes, 2),
                nn.LogSoftmax(dim=1)
            )
        self.edge_predictor = nn.Sequential(
                nn.Linear(2 * nout_gru, nhid_edges),
                nn.ReLU(),
                nn.Linear(nhid_edges, 2),
                nn.LogSoftmax(dim=1)
            )
            
        self.attention = nn.Sequential(
                nn.Linear(2 * nout_gru, 1),
                nn.Sigmoid()
            )
        
        if cuda:
            self.bn = nn.BatchNorm1d(self.nout_gru).cuda()
        else:
            self.bn = nn.BatchNorm1d(self.nout_gru)
        
        
    def get_node_an_scores(self, full_adj, final_emb):
        """
        Get anomaly scores for nodes given their embeddings
        and the adjacency matrix in the last snapshot
        """
        # Remove diagonal from adjacency matrix (i.e. self-loops)
        idx = full_adj.coalesce().indices()
        val = full_adj.coalesce().values()
        filtered_idx = idx[:,idx[0] != idx[1]]
        deg = torch.bincount(filtered_idx[0], minlength=full_adj.shape[0])
        diag = torch.where(deg != 0, 1/deg, 1)
        
        # Learn attention weights for neighbors
        source_nodes = final_emb[filtered_idx[0]]
        neigh_embs = final_emb[filtered_idx[1]]
        weights = self.attention(torch.cat((source_nodes, neigh_embs), dim=1))

        # Create adjacency matrix where the edge weights are the attention weights
        # normalized by the node degree
        adj = torch.sparse.FloatTensor(
            filtered_idx,
            diag[filtered_idx[0]] * weights.squeeze(),
            full_adj.shape
        )

        # Predict node anomaly score
        neighborhood_emb = torch.sparse.mm(adj, final_emb)
        pred = self.node_predictor(torch.cat([final_emb, neighborhood_emb], dim=1))
        return pred, weights

    
    def forward(self, start_day, end_day, adjs, edges, ret_emb=False, pert_adj=None):
        """
        PARAMS: 
        
        start_day: initial snapshot of history to consider
        end_day: last snapshot of history (i.e. final embeddings will be at snapshot end_day + 1)
        adjs: list of adjacency matrixes (one per snapshot)
        ret_emb: if True, return node predictions, final and historical node embeddings and attention weights 
                if False, return node predictions
        """
        out = torch.zeros([end_day-start_day+1,self.n_nodes,self.nout])
        # Set feature matrix to identity
        diag_feat = torch.sparse.FloatTensor(
                            torch.stack([torch.arange(0,self.n_nodes),
                            torch.arange(0,self.n_nodes)],dim=0), 
                            torch.ones([self.n_nodes, ]), (self.n_nodes, self.n_nodes))
        
        if self.is_cuda:
            out = out.cuda()
            diag_feat = diag_feat.cuda()
        
        # Apply GCN on each snapshot separately
        for i in range(end_day-start_day+1):
            out[i] = self.gcn(diag_feat, adjs[start_day + i].float())
    
        # Forward node embeddings through the GRU
        self.gru.flatten_parameters()
        _, final_emb = self.gru(out)
        final_emb = final_emb.squeeze()
        final_emb = self.bn(final_emb)
        
        # Predict edge scores
        nodes_first = final_emb[edges[0,:].long(), :]
        nodes_second = final_emb[edges[1,:].long(), :]
        pred_edges = self.edge_predictor(torch.cat([nodes_first, nodes_second], dim=1))

        # Predict node scores
        pred_nodes, weights_nodes = self.get_node_an_scores(adjs[end_day+1], final_emb)
        if pert_adj is not None:
            pred1_nodes, weights1_nodes = self.get_node_an_scores(pert_adj, final_emb)
            pred_nodes = torch.cat([pred_nodes, pred1_nodes])
            weights_nodes = torch.cat([weights_nodes, weights1_nodes])
        
        if ret_emb:
            return pred_edges, pred_nodes, out, final_emb, weights_nodes
        
        return pred_edges, pred_nodes
