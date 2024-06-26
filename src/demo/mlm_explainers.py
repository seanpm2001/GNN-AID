

def subgraphx():
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from dig.xgraph.dataset import SynGraphDataset
    from dig.xgraph.method import SubgraphX
    from dig.xgraph.method.subgraphx import PlotUtils
    from dig.xgraph.method.subgraphx import find_closest_node_result

    from dig.xgraph.method.shapley import MarginalSubgraphDataset
    # FIXME Monkey Patch for SubgraphX until DIG library doesn't support torch-geometric 2.3.1
    # PATCH BEGIN
    MarginalSubgraphDataset.__abstractmethods__ = frozenset()
    # PATCH END

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = SynGraphDataset('./datasets', 'BA_shapes')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]

    # Model
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, *args, **kwargs):
            x, edge_index, batch = self.arguments_read(*args, **kwargs)

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

        @staticmethod
        def arguments_read(*args, **kwargs):
            """
            The method is launched when the forward is executed extracts from the variable data or kwargs
            the data necessary to pass the forward: x, edge_index, batch

            !! ATTENTION: Must not be changed !!

            :param args:
            :param kwargs:
            :return: x, edge_index, batch: TORCH TENSORS
            """
            data = kwargs.get('data') or None

            if not data:
                if not args:
                    assert 'x' in kwargs
                    assert 'edge_index' in kwargs
                    x, edge_index = kwargs['x'], kwargs['edge_index'],
                    batch = kwargs.get('batch')
                    if batch is None:
                        batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64,
                                            device=x.device)
                elif len(args) == 2:
                    x, edge_index = args[0], args[1]
                    batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
                elif len(args) == 3:
                    x, edge_index, batch = args[0], args[1], args[2]
                else:
                    raise ValueError(
                        f"forward's args should take 2 or 3 arguments but got {len(args)}")
            else:
                x, edge_index, batch = data.x, data.edge_index, data.batch

            return x, edge_index, batch

    model = GCN().to(device)

    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

    # Explainer
    explainer = SubgraphX(model, num_classes=4, device=device,
                          explain_graph=False, reward_method='nc_mc_l_shapley')

    max_nodes = 5
    node_idx = 515
    print(f'explain graph node {node_idx}')
    logits = model(data.x, data.edge_index)
    prediction = logits[node_idx].argmax(-1).item()

    _, explanation_results, related_preds = explainer(data.x, data.edge_index, node_idx=node_idx,
                                                      max_nodes=max_nodes)
    explanation_results = explanation_results[prediction]
    explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)

    # Visualization
    plotutils = PlotUtils(dataset_name='ba_shapes', is_show=True)
    explainer.visualization(explanation_results,
                            max_nodes=max_nodes,
                            plot_utils=plotutils,
                            y=data.y)

    tree_node_x = find_closest_node_result(explanation_results, max_nodes=max_nodes)
    mapping = {k: int(v) for k, v in enumerate(explainer.mcts_state_map.subset)}
    tree_node_x.coalition = [mapping[k] for k in tree_node_x.coalition]
    print("CRITICAL SUBGRAPH:", tree_node_x.coalition)


def neural_analysis():
    import numpy as np
    import copy
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.nn import Linear
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    from demo.na.concepts import ConceptSet

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = TUDataset(root='./datasets', name='MUTAG')
    # loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Model
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            self.conv2 = GCNConv(16, 8)
            self.lin = Linear(8, dataset.num_classes)
            self._partial = False

        def forward(self, *args, **kwargs):
            x, edge_index, batch = self.arguments_read(*args, **kwargs)

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)

            if self._partial:
                return x

            # Readout layer
            x = global_mean_pool(x, torch.tensor([0]) if batch is None else batch)
            x = F.dropout(x, training=self.training)
            x = self.lin(x)
            return F.log_softmax(x, dim=1)

        def partial_forward(self, *args, **kwargs):
            self._partial = True
            layer_emb_dict = self(*args, **kwargs)
            self._partial = False
            return layer_emb_dict

        @staticmethod
        def arguments_read(*args, **kwargs):
            """
            The method is launched when the forward is executed extracts from the variable data or kwargs
            the data necessary to pass the forward: x, edge_index, batch

            !! ATTENTION: Must not be changed !!

            :param args:
            :param kwargs:
            :return: x, edge_index, batch: TORCH TENSORS
            """
            data = kwargs.get('data') or None

            if not data:
                if not args:
                    assert 'x' in kwargs
                    assert 'edge_index' in kwargs
                    x, edge_index = kwargs['x'], kwargs['edge_index'],
                    batch = kwargs.get('batch')
                    if batch is None:
                        batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64,
                                            device=x.device)
                elif len(args) == 2:
                    x, edge_index = args[0], args[1]
                    batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
                elif len(args) == 3:
                    x, edge_index, batch = args[0], args[1], args[2]
                else:
                    raise ValueError(
                        f"forward's args should take 2 or 3 arguments but got {len(args)}")
            else:
                x, edge_index, batch = data.x, data.edge_index, data.batch

            return x, edge_index, batch
    model = GCN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model.train()
    for epoch in range(200):
        for data in train_loader:
            # for batch in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            batch_loss = F.nll_loss(out, data.y)
            batch_loss.backward()
            optimizer.step()

    model.eval()
    test_loader = DataLoader(test_dataset)
    correct = 0
    for data in test_loader:
        pred = model(data.x, data.edge_index).argmax(dim=1)
        correct += pred == data.y
    acc = int(correct) / len(test_dataset)
    print(f'Accuracy: {acc:.4f}')

    # Explainer
    def edge_index_to_tuples(edge_index):
        return [(pair[0].item(), pair[1].item()) for pair in edge_index.T]

    def add_edge(g, u, v):
        if (u, v) in edge_index_to_tuples(g.edge_index):
            return
        g.edge_index = torch.column_stack([g.edge_index, torch.tensor([[u, v], [v, u]])])

    def concept_search(level=0, depth=3, neuron_idxs=None, top=64, augment=False, omega=None, **kwargs):
        if omega is None:
            omega = [10, 20, 20]

        assert depth >= 1 and ((level is None) or (neuron_idxs is None))

        dataset_aug = []
        if augment:
            print('Graph augmentation')

            for graph in dataset:
                edge_tuples = edge_index_to_tuples(graph.edge_index)
                dataset_aug.append(copy.deepcopy(graph))
                for _ in range(4):
                    graph_new = copy.deepcopy(graph)
                    for _ in range(8):
                        node_i = np.random.randint(0, graph_new.x.shape[0])
                        node_j = np.random.randint(0, graph_new.x.shape[0])
                        add_edge(graph_new, node_i, node_j)
                    dataset_aug.append(graph_new)
            new_dataset = dataset_aug
        else:
            new_dataset = dataset

        n_graphs = len(new_dataset)
        concept_set = ConceptSet(new_dataset, 'MUTAG', omega=omega)

        print('Performing inference')
        neuron_activations = []
        graph_sizes = []
        graph_inds = []

        for i in range(n_graphs):
            graph = new_dataset[i]
            feature_maps = model.partial_forward(x=graph.x, edge_index=graph.edge_index).detach().cpu().T
            neuron_activations.append(feature_maps)
            graph_sizes.extend([graph.x.shape[0]] * graph.x.shape[0])
            graph_inds.extend([i] * graph.x.shape[0])

        print('Keeping only top neurons')
        neuron_activations = torch.cat(neuron_activations, 1)
        nrns_vals = (neuron_activations != 0).sum(axis=1)
        neuron_idxs = nrns_vals.argsort()
        non_zero_neuron_idxs = []
        for idx in neuron_idxs:
            if nrns_vals[idx] == 0:
                continue
            non_zero_neuron_idxs.append(idx)
        non_zero_neuron_idxs = torch.LongTensor(non_zero_neuron_idxs)
        neuron_idxs = non_zero_neuron_idxs
        neuron_activations = neuron_activations.index_select(0, neuron_idxs[-top:])

        print('Performing search')

        for i in range(depth):
            ret = concept_set.match(neuron_activations, torch.tensor(graph_sizes), torch.tensor(graph_inds))

            if i < depth - 1:
                print('Adding concepts')
                concept_set.expand()
                print('Number of concepts: ' + str(concept_set.num_concepts()))

        concept_set.free()

        ret_dic = {}
        for k, v in ret.items():
            ret_dic[neuron_idxs[k].item()] = v

        return ret_dic

    def clean_concepts(neuron_concepts):
        cleaned_concepts = {}
        for neuron_idx, dic in neuron_concepts.items():
            top = sorted([(k, v) for (k, v) in dic.items()], key=lambda x: -x[1][1])
            val = top[0][1][1]
            i = 0
            best_obj = th = None
            best_obj_name = ''
            while i < len(top) and top[i][1][1] == val:
                if best_obj is None or top[i][1][0].length() < best_obj.length():
                    best_obj = top[i][1][0]
                    th = top[i][1][2]
                    best_obj_name = top[i][0]
                i += 1
            cleaned_concepts[neuron_idx] = (best_obj, (val, th, best_obj_name))
        cleaned_concepts = {k: v for (k, v) in
                            sorted(list(cleaned_concepts.items()), key=lambda x: x[1][1],
                                   reverse=True)}
        distilled = []
        for k, v in cleaned_concepts.items():
            if v[0] is None:
                continue
            if not any([v[0].name() == conc.name() for conc in distilled]):
                distilled.append(v[0])
        return cleaned_concepts, distilled

    neuron_concepts = concept_search()
    cleaned_concepts, distilled = clean_concepts(neuron_concepts)
    for neuron, concept in sorted(cleaned_concepts.items()):
        print(neuron, ':', concept[0])


if __name__ == '__main__':
    subgraphx()

    neural_analysis()

    pass
