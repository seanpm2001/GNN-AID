import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


class GNNLinear(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super(GNNLinear, self).__init__()

        # Initialize the layers
        self.conv1 = GCNConv(num_features, hidden, add_self_loops=False, bias=False)
        self.conv2 = GCNConv(hidden, num_classes, add_self_loops=False, bias=False)

    def forward(self, x=None, edge_index=None, **kwargs):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def data_to_csr_matrix(data):

    # Create sparse matrix CSR for edges
    adj_tensor = data.edge_index

    num_edges = adj_tensor.size(1)
    num_vertices = data.x.size(0)

    # Dividing a tensor into rows and columns
    rows = adj_tensor[0].numpy()
    cols = adj_tensor[1].numpy()

    # Edge weights (default 1)
    data_edges = [1] * num_edges

    # Creating a Sparse CSR Matrix
    adj_matrix = sp.csr_matrix((data_edges, (rows, cols)), shape=(num_vertices, num_vertices))

    # Create sparse matrix CSR for nodes
    attr_matrix = sp.csr_matrix(data.x.numpy())
    labels = data.y.numpy()

    return adj_matrix, attr_matrix, labels


def learn_w1_w2(dataset):
    data = dataset.dataset.data
    # TODO передавать параметр hidden
    model_gnn_lin = GNNLinear(dataset.num_node_features, 16, dataset.num_classes)

    optimizer = torch.optim.Adam(model_gnn_lin.parameters(),
                                 lr=0.001,
                                 weight_decay=5e-4)

    num_epochs = 2000
    print("Train surrogate model")
    for epoch in tqdm(range(num_epochs)):
        model_gnn_lin.train()
        optimizer.zero_grad()
        preds = model_gnn_lin(data.x, data.edge_index)
        loss = F.cross_entropy(preds[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    print("End training")

    W1 = model_gnn_lin.conv1.lin.weight.T
    W2 = model_gnn_lin.conv2.lin.weight.T
    return W1, W2



