# Detecting Edge and Node Anomalies with Temporal GNNs

Code for the paper "Detecting Edge and Node Anomalies with Temporal GNNs", Proceedings of the 3rd GNNet Workshop@CoNEXT 2024.

## Table of Content
1) [Repository structure](#structure)<br>
2) [Datasets](#datasets)<br>
3) [Usage](#usage)<br>

## Repository structure<a name="structure"></a>
This repository contains the code to implement GCN-GRU for anomaly detection on nodes and edges on graph data and the four real-world datasets with injected anomalies used in the paper. The code is organized as follows.

```
gcn-gru/
  +-- scripts/
  |     +-- preprocessing/
  |     |    +-- preprocessing.py
  |     +-- tgnn/
  |     |    +-- gcngru.py
  |     |    +-- models.py
  |     +-- utils/
  |     |    +-- utils.py
  +-- notebooks/
  |     +-- demo.ipynb  
  |     ...
  +-- data/
  |     ...
```

#### Scripts<a name="scripts"></a>
- `preprocessing.py`: functions to preprocess data
- `gcngru.py`: wrapper class for the base models
- `models.py`: description of base models (GCN, GCN-GRU for nodes, edges and both)
- `utils.py`: utility functions

#### Notebooks<a name="notebooks"></a>
- `demo.ipynb`: example of a single training and testing for anomaly detection (node-only, edge-only and both)

#### Data<a name="data"></a>

- Each file named `adjs_anom_dataSet` is a list of matrices (one per snapshot). Each matrix contains original edges + injected anomalies. They represent both the Graph and the "Features". 
- Each file named `anomalies_edges_idx_dataSet` is a list of boolean arrays (one per snapshot). True means that the edge is *anomalous*, False means that the edge is *normal*. They represent the EDGE ground truth
- Each file named `anomalies_nodes_idx_dataSet` is a list of boolean arrays (one per snapshot). True means that the node is *anomalous*, False means that the node is *normal*. They represent the NODE ground truth



## Datasets<a name="datasets"></a>

|               | Bipartite |                          Docs                          |       Event       |
|---------------|:---------:|:------------------------------------------------------:|:-----------------:|
| `reddit`        |     Y     | [Reddit](https://snap.stanford.edu/data/web-RedditNetworks.html) |   Social posting  |
| `webbrowsing`   |     Y     |                          [WebBrowsing](#webbrowsing_dset)                          |    Web browsing   |
| `stackoverflow` |     N     |  [StackOverflow](https://snap.stanford.edu/data/sx-stackoverflow.html)  |   Community interaction  |
| `uci` |     N     |  [UCI](https://github.com/yuetan031/TADDY_pytorch/tree/main/data/raw)  |   Messages on social network  |


## Usage<a name="usage"></a>


#### Perform a single experiment<a name="run_exp"></a>

The notebook `demo` allows to perform a single training and test experiment. To use it, specify the desired dataset and the model parameters. The results are printed and the anomaly scores for edges and nodes are saved.

## Notes
In `demo.ipynb`, the variable `splits` is a tuple with 5 variables. They are:
- `history`: number of snapshots used as history
- `train_start`: first training snapshot ID -1
- `train_end`: last training snapshot ID
- `val`: number of snapshots used as validation 
- `test`: final snapshot.
E.g.:
```
splits = (10, 9, 19, 5, 29)
```
this means that 
- the history starts at $t_0$ and ends at $t_9$
- the training starts at $t_{10}$ and ends at $t_{19}$
- the validation starts at $t_{20}$ and ends at $t_{24}$
- the test starts at $t_{25}$ and ends at $t_{29}$
