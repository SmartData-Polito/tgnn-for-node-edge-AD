from .models import GCNGRU_nodes, GCNGRU, GCNGRU_both
import torch
from tqdm import tqdm
from ..utils import perturb_adjacency, compute_accuracy
from ..preprocessing import get_self_supervised_edges
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import os
import shutil

class GCN_GRU():
    def __init__(self, entities='nodes', cuda=False, epochs=None, ns=1, ns_edge=1,
            splits=(5, 14, 86, 500), n=None, LAMBDA=.5, bipartite=False,
                nout=512, nout_gru=512, nhid_gcn=1024, nhid_nodes=1024, nhid_edges=1024,
                dropout=.0, n_layers=1,lr=0.001):
        """ Description

        Parameters
        ----------
        entities : str, optional
            Type of entities to consider ('nodes', 'edges', 'both').
        cuda : bool, optional
            Whether to use CUDA for computation.
        epochs : int or None, optional
            Number of epochs for training. If None, no limit is set.
        ns : int, optional
            Number of negative samples.
        splits : tuple, optional
            Tuple containing the number of days to consider as history, training set, 
            validation set, and test set.
        n : int or None, optional
            Number of nodes (relevant only if entities is 'nodes' or 'both').
        """
        # General parameters initializations
        self.epochs, self.ns, self.entities, self.n, self.bipartite, self.ns_edge = epochs, ns, entities, n, bipartite, ns_edge
        self.cuda = cuda

        # Each entry of the tuple is the number of days to consider as 
        # history, training set, validation set, test set
        self.hist, self.train_start, self.train_end, self.val, self.test = splits

        # Initialize the model
        if self.entities == 'nodes':
            self.model = GCNGRU_nodes(nout=nout, 
                                      nout_gru=nout_gru,
                                      nhid_gcn=nhid_gcn, 
                                      nhid_nodes=nhid_nodes,
                                      dropout=dropout, 
                                      n_layers=n_layers, 
                                      n_nodes=self.n,
                                      cuda=cuda)
        elif self.entities == 'edges':
            self.model = GCNGRU(nout=nout, 
                                nout_gru=nout_gru,
                                nhid_gcn=nhid_gcn, 
                                nhid_edges=nhid_edges, 
                                dropout=dropout, 
                                n_layers=n_layers, 
                                n_nodes=self.n,
                                cuda=cuda)
        elif self.entities == 'both':
            self.LAMBDA = LAMBDA
            self.model = GCNGRU_both(nout=nout,
                                nout_gru=nout_gru, 
                                nhid_gcn=nhid_gcn, 
                                nhid_edges=nhid_edges, 
                                nhid_nodes=nhid_nodes, 
                                dropout=dropout, 
                                n_layers=n_layers, 
                                n_nodes=self.n,
                                cuda=cuda)
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=lr, 
                                          weight_decay=5e-3)
        # Manage CUDA if available and specified
        if self.cuda:
            self.model = self.model.cuda()

    def _train_single_step_nodes(self, X, Xn, it, n_days):
        """ Perform a single training step for nodes.

        Parameters
        ----------
        X : list of torch.Tensor
            Input data.
        Xn : list of torch.Tensor
            List of perturbed input data.
        it : int
            Current timestep.
        n_days : int
            Total number of days.

        Returns
        -------
        train_loss : torch.Tensor
            Training loss.
        accuracy : float
            Accuracy of the predictions.
        y_pred : torch.Tensor
            Predicted values.

        """
        
        start_day = n_days - self.hist
        self.optimizer.zero_grad()   
        
        # Find active nodes (maybe this could be done somewhere else)
        index = X[n_days + 1].coalesce().indices()
        active_nodes = index[0, index[0] != index[1]].unique()
        
        # Get predictions
        y_pred  = [self.model(start_day, n_days, 
                              X, pert_adj=Xn[0][it])]
        y_pred[0] = torch.cat([y_pred[0][active_nodes], y_pred[0][self.n:][active_nodes]])
        y_pred += [self.model(start_day, n_days, 
                              X, pert_adj=x[it])[self.n:][active_nodes] for x in Xn[1:]]
        y_pred  = torch.cat(y_pred)

        # Retrieve self-supervised true labels
        y_true = torch.ones([y_pred.shape[0],]).long()
        y_true[len(active_nodes):] = 0

        if self.cuda:
            y_true = y_true.cuda()
        # Compute loss and do a backward pass
        train_loss = F.nll_loss(y_pred, 
                                y_true)
            
        train_loss.backward()
        self.optimizer.step()

        # Compute 'local' accuracy
        _y_pred = torch.exp(y_pred[:,1].view(-1,))
        accuracy = compute_accuracy(_y_pred, y_true)

        return train_loss, accuracy, y_pred
    
    def _train_single_step_edges(self, X, Xn, it, n_days):
        """ Perform a single training step for edges.

        Parameters 
        ----------
        X : Tuple with
            X = (adjs, idx_test, X_train, index)
        Xn : Xn is neg_edges
        it : int
            Current timestep.
        n_days : int
            Total number of days.

        Returns
        -------
        train_loss : torch.Tensor
            Training loss.
        accuracy : float
            Accuracy of the predictions.
        y_pred : torch.Tensor
            Predicted values.

        """
        start_day = n_days - self.hist
        self.optimizer.zero_grad()   
        
        # Get predictions
        X, index = X
        y_pred = self.model(start_day, n_days, 
                            X, torch.cat([index[:, index[0] < index[1]], Xn], dim=1))

        # Retrieve self-supervised true labels
        y_true = torch.cat([torch.ones([Xn.shape[1] // self.ns_edge,]), 
                            torch.zeros([Xn.shape[1],])]).long()

        if self.cuda:
            y_true = y_true.cuda()
        
        # Compute loss 
        train_loss = F.nll_loss(y_pred, y_true)

        train_loss.backward()
        self.optimizer.step()

        # Compute 'local' accuracy
        _y_pred = torch.exp(y_pred[:,1].view(-1,))
        accuracy = compute_accuracy(_y_pred, y_true)

        return train_loss, accuracy, y_pred
    
    def _train_single_step_both(self, X, Xn, it, n_days):
        """ Perform a single training step for nodes and edges.

        Parameters 
        ----------
        X : Tuple with
            X = (adjs, idx_test, index)
        Xn : tuple of torch.Tensor (Xn_nodes, Xn_edges)
            Xn_nodes is the perturbed adjacency matrix
            Xn_edges is the list of negative edges            
        it : int
            Current timestep.
        n_days : int
            Total number of days.

        Returns
        -------
        train_loss : torch.Tensor
            Training loss.
        accuracy : Tuple (accuracy edges, accuracy nodes)
            Accuracy of the predictions.
        y_pred : Tuple of torch.Tensor
            (predictions edges, predictions nodes)
            Predicted values.

        """
        start_day = n_days - self.hist
        self.optimizer.zero_grad()   

        Xn_edges, Xn_nodes = Xn
        X, index = X
        active_nodes = index[0, index[0] != index[1]].unique()
        y_pred_edges, y_pred_nodes = self.model(start_day, n_days, X, 
                            torch.cat([index[:, index[0] < index[1]], Xn_edges], dim=1), pert_adj=Xn_nodes[it])
        y_pred_nodes = torch.cat([y_pred_nodes[active_nodes], y_pred_nodes[self.n:][active_nodes]])
        
        # Compute loss and accuracy
        # Nodes
        train_labels_nodes = torch.ones([y_pred_nodes.shape[0],]).long()
        train_labels_nodes[len(active_nodes):] = 0
        # Edges
        train_labels_edges = torch.ones([y_pred_edges.shape[0],]).long()
        train_labels_edges[Xn_edges.shape[1]:] = 0
        
        if self.cuda:
            train_labels_nodes = train_labels_nodes.cuda()
            train_labels_edges = train_labels_edges.cuda()
        # Update
        train_loss_nodes = F.nll_loss(y_pred_nodes, train_labels_nodes)
        train_loss_edges = F.nll_loss(y_pred_edges, train_labels_edges)
        train_loss = self.LAMBDA * train_loss_nodes + (1 - self.LAMBDA) * train_loss_edges
            
        train_loss.backward()
        self.optimizer.step()

        # Compute 'local' accuracy        
        _y_pred_nodes = torch.exp(y_pred_nodes[:,1].view(-1,))
        _y_pred_edges = torch.exp(y_pred_edges[:,1].view(-1,))
            
        accuracy_nodes = compute_accuracy(_y_pred_nodes, train_labels_nodes)
        accuracy_edges = compute_accuracy(_y_pred_edges, train_labels_edges)

        return train_loss, (accuracy_edges, accuracy_nodes), (y_pred_edges, y_pred_nodes)
    
    
    def _evaluate_single_step_both(self, X, n_days, ret_emb=False, dir_name=None):
        """ Perform a single evaluation step for nodes&edges.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        n_days : int
            Total number of days.

        Returns
        -------
        y_pred : torch.Tensor
            Predicted values.
        """
        start_day = n_days - self.hist
        self.model.eval()

        index = X[n_days+1].coalesce().indices()
        active_nodes = index[0, index[0] != index[1]].unique()
        if ret_emb:
            y_pred_edges, y_pred_nodes, out, final_emb, weights = self.model(start_day, n_days, X, index[:, index[0] < index[1]], ret_emb=True)
            torch.save(final_emb, dir_name + f"/final_emb_{n_days}")
            torch.save(weights, dir_name + f"/weights_{n_days}")
        else:
            y_pred_edges, y_pred_nodes = self.model(start_day, n_days, X, index[:, index[0] < index[1]])
        y_pred_edges, y_pred_nodes = y_pred_edges[:,1], y_pred_nodes[active_nodes,1]

        return (y_pred_edges, y_pred_nodes)

    def _evaluate_single_step(self, X, n_days, ret_emb=False, dir_name=None):
        """ Perform a single evaluation step for nodes or edges.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        n_days : int
            Total number of days.
   
        Returns
        -------
        y_pred : torch.Tensor
            Predicted values.
        """
        start_day = n_days - self.hist
        self.model.eval()

        if self.entities == 'nodes':
            index = X[n_days+1].coalesce().indices()
            active_nodes = index[0, index[0] != index[1]].unique()
            if ret_emb:
                y_pred, _, final_emb, _ = self.model(start_day, n_days, X, ret_emb=True)
                y_pred = y_pred[active_nodes,1]
                torch.save(final_emb, dir_name + f"/final_emb_{n_days}")
            else:
                y_pred = self.model(start_day, n_days, X)[active_nodes,1]

        if self.entities == 'edges':
            index = X[n_days+1].coalesce().indices()
            if ret_emb:
                y_pred, _, final_emb = self.model(start_day, n_days, X, 
                                index[:, index[0] < index[1]], ret_emb=True)
                y_pred = y_pred[:,1]
                torch.save(final_emb, dir_name + f"/final_emb_{n_days}")
            else:
                y_pred = self.model(start_day, n_days, X, 
                                index[:, index[0] < index[1]])[:,1]

        return y_pred

    def fit(self, X, y_val=None, save=False, dset=None, dir_name=None):
        """
        Fit the model using training data. 
        To save the model and the training/validation results, 
        set save=True and specify the output directory in dir_name
        """
        if save:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            with open(dir_name + "/training_history.csv", "a") as f:
                if self.entities in ["nodes", "edges"]:
                    f.write(f"Epoch,Train_loss,Train_acc,Val_auc")
                elif self.entities == "both":
                    f.write(f"Epoch,Train_loss,Train_acc_edges,Train_acc_nodes,Val_auc_edges,Val_auc_nodes")
        train_range = range(self.train_start, self.train_end)
        eval_range = range(self.train_end, self.train_end + self.val)
        delta_t = self.train_end - self.train_start
        
        # X is the sparse adjacency matrix
        if self.cuda:
            X = [x.cuda() for x in X]

        best_tot_val, best_val_edges_nodes, best_train_acc, best_model, best_val_true_pred = 0.0, None, None, None, None
        # Get negative pairs
        if self.entities == 'nodes' or self.entities == 'both':
            X_neg = [perturb_adjacency(X, 
                                    self.train_start, 
                                    self.train_end,
                                    self.bipartite,
                                    cuda=self.cuda) for i in range(self.ns)]
        for epoch in range(self.epochs):
            # General init at the beginning of each epoch
            if self.entities == 'both':
                tot_acc_nodes, tot_acc_edges, tot_loss = .0, .0, .0
            else:
                tot_acc, tot_loss = .0, .0
            self.model.train()

            # ----------------------------------------------------------------
            # Training steps
            # ----------------------------------------------------------------
            pbar = tqdm(train_range, desc=f'Epoch {epoch}')
            for i, n_days in enumerate(pbar):
                # Do a single training pass
                if self.entities == 'nodes':
                    train_loss, train_acc, y_pred = self._train_single_step_nodes(X, 
                                                                X_neg, i, n_days)
                    # Compute training metrics
                    tot_loss += train_loss.detach().item() / (delta_t)
                    tot_acc  += train_acc.detach().item() / (delta_t)
                    
                elif self.entities == 'edges':
                    X_neg, index = get_self_supervised_edges(X, 
                                                    n_days, cuda=self.cuda, 
                                                    bipartite=self.bipartite, ns_edge=self.ns_edge)
                    _X_ = (X, index)
                    train_loss, train_acc, y_pred = self._train_single_step_edges(_X_, 
                                                                X_neg, i, n_days)
                    # Compute training metrics
                    tot_loss += train_loss.detach().item() / (delta_t)
                    tot_acc  += train_acc.detach().item() / (delta_t)
                    
                elif self.entities == 'both':
                    X_neg_edges, index = get_self_supervised_edges(X, 
                                                    n_days, cuda=self.cuda, 
                                                bipartite=self.bipartite, ns_edge=self.ns_edge)
                    _Xn_ = (X_neg_edges, X_neg[0])
                    _X_ = (X, index)
                    train_loss, (train_acc_edges, train_acc_nodes), (y_pred_edges, y_pred_nodes) = \
                        self._train_single_step_both(_X_, _Xn_, i, n_days)
                    # Compute training metrics
                    tot_loss += train_loss.detach().item() / (delta_t)
                    tot_acc_edges  += train_acc_edges.detach().item() / (delta_t)
                    tot_acc_nodes  += train_acc_nodes.detach().item() / (delta_t)
                    
            # ----------------------------------------------------------------
            # Evaluation steps
            # ----------------------------------------------------------------
            with torch.no_grad():
                if self.entities == 'edges' or self.entities == 'nodes':

                    # Get predictions
                    y_pred = [self._evaluate_single_step(X, n_days) \
                                      for i, n_days in enumerate(eval_range)]

                    # Compute validation metrics
                    val_auc = roc_auc_score(y_val.cpu().numpy(), 
                                            torch.cat(y_pred).view(-1,1).cpu().numpy())
                    print(f"\tTrain loss: {round(tot_loss, 4)} - "\
                          f"Train acc: {round(tot_acc, 4)} - "\
                          f"Val auc: {round(val_auc, 4)}")
                    if save:
                        with open(dir_name + "/training_history.csv", "a") as f:
                            f.write(f"\n{epoch},{round(tot_loss, 4)},{round(tot_acc, 4)},{round(val_auc, 4)}")

                elif self.entities == 'both':
                    # Get predictions
                    y_pred_edges, y_pred_nodes = [], []
                    for i, n_days in enumerate(eval_range):
                        y_pred_edges_i, y_pred_nodes_i = self._evaluate_single_step_both(X, n_days)
                        y_pred_edges.append(y_pred_edges_i)
                        y_pred_nodes.append(y_pred_nodes_i)

                    # Compute validation metrics
                    y_val_edges, y_val_nodes = y_val
                    val_auc_edges = roc_auc_score(y_val_edges.cpu().numpy(), 
                                            torch.cat(y_pred_edges).view(-1,1).cpu().numpy())
                    val_auc_nodes = roc_auc_score(y_val_nodes.cpu().numpy(), 
                                            torch.cat(y_pred_nodes).view(-1,1).cpu().numpy())
                    print(f"\tTrain loss: {round(tot_loss, 4)} - "\
                          f"Train acc edges: {round(tot_acc_edges, 4)} - "\
                          f"Train acc nodes: {round(tot_acc_nodes, 4)} - "\
                          f"Val auc edges: {round(val_auc_edges, 4)} - "\
                          f"Val auc nodes: {round(val_auc_nodes, 4)}")
                    if save:
                        with open(dir_name + "/training_history.csv", "a") as f:
                            f.write(f"\n{epoch},{round(tot_loss, 4)},{round(tot_acc_edges, 4)},{round(tot_acc_nodes, 4)},{round(val_auc_edges, 4)},{round(val_auc_nodes, 4)}")
                    
                    val_auc = val_auc_nodes + val_auc_edges # Used for model selection
                    
                    
            # Take best performing model
            if val_auc > best_tot_val: 
                best_tot_val = val_auc
                if self.entities == 'both':
                    best_val_edges_nodes = (val_auc_nodes, val_auc_edges)
                    best_train_acc = (tot_acc_nodes, tot_acc_edges)
                    best_val_true_pred = (y_val_edges, y_val_nodes, y_pred_edges, y_pred_nodes)
                else:
                    best_val_true_pred = (y_val, y_pred)
                best_model = deepcopy(self.model)

        if save:
            # Save best model and best results for validation results
            torch.save(best_model.state_dict(), dir_name + "/best_model")
            torch.save(best_val_true_pred, dir_name + "/val_true_pred")
            self.model.load_state_dict(torch.load(dir_name + "/best_model"))
        else:
            self.model = best_model
        if self.entities == 'both':
            return best_val_edges_nodes, best_train_acc
        
    def predict(self, X, y=None, save=False, dir_name=None, ret_emb=False, dir_name_emb=None):
        """
        Perform predictions on the test data. 
        To save the predictions, set save=True,
        y=test_labels and specify the output 
        directory in dir_name
        """

        # Manage test days range
        test_range = range(self.train_end + self.val, self.test) 
        if self.cuda:
            X = [x.cuda() for x in X]
        # Evaluate
        with torch.no_grad():
            if self.entities in ["edges", "nodes"]:  
                y_pred = [self._evaluate_single_step(X, n_days, ret_emb=ret_emb, dir_name=dir_name_emb) \
                                            for i, n_days in enumerate(test_range)]
                y_pred = torch.cat(y_pred).view(-1,1)
            
            elif self.entities == "both":
                y_pred_edges, y_pred_nodes = [], []
                for i, n_days in enumerate(test_range):
                    y_pred_edges_i, y_pred_nodes_i = self._evaluate_single_step_both(X, n_days, ret_emb=ret_emb, dir_name=dir_name_emb)
                    y_pred_edges.append(y_pred_edges_i)
                    y_pred_nodes.append(y_pred_nodes_i)
                y_pred = (torch.cat(y_pred_edges).view(-1,1), torch.cat(y_pred_nodes).view(-1,1)) 
                        
        if y is not None:
            if self.entities in ["edges", "nodes"]:
                test_auc = roc_auc_score(y.numpy(), y_pred.cpu().numpy())
                print("\nTest day: Test auc: {:1.4f}".format(test_auc))
            elif self.entities == "both":
                y_edges, y_nodes = y
                test_auc_edges = roc_auc_score(y_edges.numpy(), y_pred[0].cpu().numpy())
                test_auc_nodes = roc_auc_score(y_nodes.numpy(), y_pred[1].cpu().numpy())
                print("\nTest auc edges: {:1.4f} Test auc nodes: {:1.4f}".format(test_auc_edges, test_auc_nodes))

        if save:
            torch.save((y, y_pred), dir_name + "/test_true_pred")
        
        return y_pred
