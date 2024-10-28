import torch
import torch.nn.functional as F
import numpy as np

from attacks.attack_base import Attacker

# Nettack imports
from src.attacks.nettack.nettack import Nettack
from src.attacks.nettack.utils import preprocess_graph, largest_connected_components, data_to_csr_matrix, train_w1_w2

# PGD imports
from attacks.evasion_attacks_collection.pgd.utils import Projection, RandomSampling
from attacks.evasion_attacks_collection.pgd.model import MatGNN
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph
from tqdm import tqdm
from torch_geometric.nn import SGConv


class EvasionAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class EmptyEvasionAttacker(EvasionAttacker):
    name = "EmptyEvasionAttacker"

    def attack(self, **kwargs):
        pass


class FGSMAttacker(EvasionAttacker):
    name = "FGSM"

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def attack(self, model_manager, gen_dataset, mask_tensor):
        gen_dataset.data.x.requires_grad = True
        output = model_manager.gnn(gen_dataset.data.x, gen_dataset.data.edge_index, gen_dataset.data.batch)
        loss = model_manager.loss_function(output[mask_tensor],
                                           gen_dataset.data.y[mask_tensor])
        model_manager.gnn.zero_grad()
        loss.backward()
        sign_data_grad = gen_dataset.data.x.grad.sign()
        perturbed_data_x = gen_dataset.data.x + self.epsilon * sign_data_grad
        perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
        gen_dataset.data.x = perturbed_data_x.detach()
        return gen_dataset


class PGDAttacker(EvasionAttacker):
    name = "PGD"

    def __init__(self,
                 element_idx=0,
                 perturb_ratio=0.5,
                 learning_rate=0.001,
                 num_iterations=100,
                 num_rand_trials=100):

        super().__init__()
        self.attack_diff = None
        self.element_idx = element_idx
        self.perturb_ratio = perturb_ratio
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_rand_trials = num_rand_trials

    def attack_old(self, model_manager, gen_dataset, mask_tensor):
        # Since the PGD attack is an attack on the graph structure, which requires optimization of the graph adjacency
        # matrix, we need to use the GCNConv graph model, implemented through matrix operations with the ability to
        # differentiate the adjacency matrix.
        hidden = model_manager.gnn.structure.layers[0]['layer']['layer_kwargs']['out_channels']
        model = MatGNN(num_features=gen_dataset.num_node_features, hidden=hidden, num_classes=gen_dataset.num_classes)

        # Copy learned matrix
        with torch.no_grad():
            model.conv0.linear.weight.copy_(model_manager.gnn.GCNConv_0.lin.weight)  # W0
            model.conv0.linear.bias.copy_(model_manager.gnn.GCNConv_0.bias)  # b0
            model.conv1.linear.weight.copy_(model_manager.gnn.GCNConv_1.lin.weight)  # W1
            model.conv1.linear.bias.copy_(model_manager.gnn.GCNConv_1.bias)  # b1

        # Convert the list of edges into an adjacency matrix
        data = gen_dataset.data
        adj_matrix = to_dense_adj(data.edge_index).squeeze(0)

        # eps - number of edges subject to perturbation
        total_edges = data.edge_index.size(1)
        eps = int(self.perturb_ratio * total_edges)

        # Adjacency Matrix Optimization Process
        # --------------- Start ---------------
        A = adj_matrix
        N = data.x.size(0)

        # M - training mask matrix
        M = torch.zeros((N, N), requires_grad=True)

        # Projection operator
        projection = Projection(eps=eps)

        model.eval()
        optimizer = torch.optim.Adam([M], lr=self.learning_rate, weight_decay=5e-4)

        # Optimization cycle
        progress_bar = tqdm(range(self.num_iterations), desc="Optimization cycle", leave=True, postfix={"Loss": 0.0})
        for t in progress_bar:
            # Perturbation of matrix A; A_pert is the perturbed matrix
            A_pert = A - A * M
            preds = model(data.x, A_pert)

            # calculate the loss
            loss = self.__attack_loss(preds, data.y)
            # print(f"iteration: {t}, loss: {loss:.4f}")
            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})

            # backpropagation of gradients
            optimizer.zero_grad()
            loss.backward()

            # Update M
            optimizer.step()

            with torch.no_grad():
                M.copy_(projection(M))
        # ---------------- End ----------------

        # Random Sampling
        random_sampling = RandomSampling(K=self.num_rand_trials,
                                         eps=eps,
                                         A=A,
                                         attack_loss=self.__attack_loss,
                                         model=model,
                                         data=data)
        M_binary = random_sampling(M)
        A_pert_binary = A - A * M_binary

        # Convert adjacency matrix to edge list
        edge_index, _ = dense_to_sparse(A_pert_binary)

        gen_dataset.data.edge_index = edge_index
        return gen_dataset

    def attack(self, model_manager, gen_dataset, mask_tensor):
        data = gen_dataset.data
        N = data.x.size(0)  # num nodes
        E = data.edge_index.size(1)  # num edges

        if gen_dataset.is_multi():
            self._attack_on_graph(model_manager, gen_dataset)
        else:
            self._attack_on_node(model_manager, gen_dataset)

        # edge_index = [[i for i in range(2708)], [i for i in range(2708)]]
        # edge_index = torch.tensor(edge_index)
        edge_index_can_remove = data.edge_index
        edge_index_can_add = self.generate_edge_index_can_add(data.edge_index, N)
        edge_index_joint = torch.cat([edge_index_can_remove, edge_index_can_add], dim=1)

        mask_remove = torch.ones(E)
        mask_add = torch.zeros(edge_index_can_add.size(1), requires_grad=True)
        # mask_joint = torch.cat([mask_remove, mask_add], dim=0)

        edge_weight = torch.ones(edge_index_joint.size(1))

        # TODO если веса изначально есть то присваеваем их в edge_weight, если нет, то torch.ones
        # edge_weight = torch.ones()
        # mask = torch.rand(N, requires_grad=True)
        # mask = torch.rand((data.x.size(0), data.x.size(1)), requires_grad=True)

        model = model_manager.gnn
        # model.eval()
        optimizer = torch.optim.Adam([mask_add], lr=self.learning_rate, weight_decay=5e-4)

        total_edges = data.edge_index.size(1)
        eps = int(self.perturb_ratio * total_edges)
        print(eps)
        projection = Projection(eps=eps)

        # Optimization cycle
        progress_bar = tqdm(range(self.num_iterations), desc="Optimization cycle", leave=True, postfix={"Loss": 0.0})
        # model.train()
        from src.testing.soloviov_test.attacks.pgd_attack.try_with_my_gcnconv.run import metrics
        for t in progress_bar:
            mask_joint = torch.cat([mask_remove, mask_add], dim=0)
            edge_weight_perturbed = edge_weight * mask_joint
            preds = model(data.x, edge_index_joint, None, edge_weight_perturbed)

            # calculate the loss
            loss = self.__attack_loss(preds, data.y)
            # print(f"iteration: {t}, loss: {loss:.4f}")
            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})
            print(f"{loss:.4f}")

            optimizer.zero_grad()
            # backpropagation of gradients
            loss.backward()

            # Update mask
            optimizer.step()

            with torch.no_grad():
                mask_add.copy_(projection(mask_add))

            mask_add_bool = self.random_sampling(mask_add.clone())
            mask_remove_bool = mask_remove.bool()
            mask_joint_bool = torch.cat([mask_remove_bool, mask_add_bool], dim=0)

            edge_index_joint_filtered = edge_index_joint[:, mask_joint_bool]
            out = model(data.x, edge_index_joint_filtered)
            train_acc, test_acc, train_stats, test_stats = metrics(out, data)
            print(f"epoch: {t}, loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
            # print(
            #     f"epoch: {t}, loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}, mass: {train_stats}")

    def _attack_on_node(self, model_manager, gen_dataset):
        node_idx = self.element_idx

        data = gen_dataset.data
        model = model_manager.gnn

        num_hops = model.n_layers
        N = data.x.size(0)  # num nodes

        _, edge_index_can_remove, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=False)

        edge_index_can_add = self.generate_edge_index_can_add(edge_index_can_remove, N)
        edge_index_joint = torch.cat([edge_index_can_remove, edge_index_can_add], dim=1)

        mask_remove = torch.ones(edge_index_can_remove.size(1))
        mask_add = torch.zeros(edge_index_can_add.size(1), requires_grad=True)

        edge_weight = torch.ones(edge_index_joint.size(1))

        optimizer = torch.optim.Adam([mask_add], lr=self.learning_rate, weight_decay=5e-4)

        total_edges = data.edge_index.size(1)
        eps = int(self.perturb_ratio * total_edges)
        print(eps)
        projection = Projection(eps=eps)

        prob = model(data.x, data.edge_index)[node_idx]
        print(f"real_class:{data.y[node_idx]}, predicted_class: {torch.argmax(prob).item()}, prob: {torch.exp(torch.max(prob)).item()}")

        for t in range(self.num_iterations):
            mask_joint = torch.cat([mask_remove, mask_add], dim=0)
            edge_weight_perturbed = edge_weight * mask_joint
            pred = model(data.x, edge_index_joint, None, edge_weight_perturbed)[node_idx]

            # calculate the loss
            loss = self.__attack_loss(pred, data.y[node_idx])
            # print(f"iteration: {t}, loss: {loss:.4f}")

            optimizer.zero_grad()
            # backpropagation of gradients
            loss.backward()

            # Update mask
            optimizer.step()

            with torch.no_grad():
                mask_add.copy_(projection(mask_add))

            mask_add_bool = self.random_sampling(mask_add.clone())
            mask_remove_bool = mask_remove.bool()
            mask_joint_bool = torch.cat([mask_remove_bool, mask_add_bool], dim=0)

            edge_index_joint_filtered = edge_index_joint[:, mask_joint_bool]
            prob = model(data.x, edge_index_joint_filtered)[node_idx]
            print(f"epoch: {t}, loss: {loss:.4f}, real_class:{data.y[node_idx]}, predicted_class: {torch.argmax(prob).item()}, prob: {torch.exp(prob)}")

    def _attack_on_graph(self, model_manager, gen_dataset):
        graph_idx = self.element_idx

        data = gen_dataset.dataset[graph_idx]
        model = model_manager.gnn

        N = data.x.size(0)  # num nodes
        edge_index_can_remove = data.edge_index
        edge_index_can_add = self.generate_edge_index_can_add(edge_index_can_remove, N)
        edge_index_joint = torch.cat([edge_index_can_remove, edge_index_can_add], dim=1)

        mask_remove = torch.ones(edge_index_can_remove.size(1))
        mask_add = torch.zeros(edge_index_can_add.size(1), requires_grad=True)

        edge_weight = torch.ones(edge_index_joint.size(1))

        optimizer = torch.optim.Adam([mask_add], lr=self.learning_rate, weight_decay=5e-4)

        total_edges = data.edge_index.size(1)
        eps = int(self.perturb_ratio * total_edges)
        print(eps)
        projection = Projection(eps=eps)

        prob = model(data.x, data.edge_index)
        print(f"real_class:{data.y}, predicted_class: {torch.argmax(prob).item()}, prob: {torch.exp(torch.max(prob)).item()}")

        for t in range(self.num_iterations):
            mask_joint = torch.cat([mask_remove, mask_add], dim=0)
            edge_weight_perturbed = edge_weight * mask_joint
            pred = model(data.x, edge_index_joint, None, edge_weight_perturbed)

            # calculate the loss
            loss = self.__attack_loss(pred, data.y)
            # print(f"iteration: {t}, loss: {loss:.4f}")

            optimizer.zero_grad()
            # backpropagation of gradients
            loss.backward()

            # Update mask
            optimizer.step()

            with torch.no_grad():
                mask_add.copy_(projection(mask_add))

            mask_add_bool = self.random_sampling(mask_add.clone())
            mask_remove_bool = mask_remove.bool()
            mask_joint_bool = torch.cat([mask_remove_bool, mask_add_bool], dim=0)

            edge_index_joint_filtered = edge_index_joint[:, mask_joint_bool]
            prob = model(data.x, edge_index_joint_filtered)

            print(
                f"epoch: {t}, loss: {loss:.4f}, real_class:{data.y}, predicted_class: {torch.argmax(prob).item()}, prob: {torch.exp(prob)}")

    def try_new_attack(self, model_manager, gen_dataset):
        data = gen_dataset.data
        N = data.x.size(0)  # num nodes
        E = data.edge_index.size(1)  # num edges

        # edge_index = [[i for i in range(2708)], [i for i in range(2708)]]
        # edge_index = torch.tensor(edge_index)
        edge_index_can_remove = data.edge_index
        # edge_index_can_add = self.generate_edge_index_can_add(data.edge_index, N)
        # edge_index_joint = torch.cat([edge_index_can_remove, edge_index_can_add], dim=1)

        mask_remove = torch.ones(E, requires_grad=True)
        # mask_add = torch.zeros(edge_index_can_add.size(1), requires_grad=True)
        # mask_joint = torch.cat([mask_remove, mask_add], dim=0)

        # edge_weight = torch.ones(edge_index_joint.size(1))

        # TODO если веса изначально есть то присваеваем их в edge_weight, если нет, то torch.ones
        # edge_weight = torch.ones()
        # mask = torch.rand(N, requires_grad=True)
        # mask = torch.rand((data.x.size(0), data.x.size(1)), requires_grad=True)

        model = model_manager.gnn
        # model.eval()
        optimizer = torch.optim.Adam([mask_remove], lr=self.learning_rate, weight_decay=5e-4)

        total_edges = data.edge_index.size(1)
        eps = int(self.perturb_ratio * total_edges)
        print(eps)
        projection = Projection(eps=eps)

        # Optimization cycle
        progress_bar = tqdm(range(self.num_iterations), desc="Optimization cycle", leave=True, postfix={"Loss": 0.0})
        # model.train()
        from src.testing.soloviov_test.attacks.pgd_attack.try_with_my_gcnconv.run import metrics
        for t in progress_bar:
            # mask_joint = torch.cat([mask_remove, mask_add], dim=0)
            edge_weight_perturbed = edge_weight * mask_joint
            preds = model(data.x, edge_index_joint, None, edge_weight_perturbed)

            # calculate the loss
            loss = self.__attack_loss(preds, data.y)
            # print(f"iteration: {t}, loss: {loss:.4f}")
            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})
            print(f"{loss:.4f}")

            optimizer.zero_grad()
            # backpropagation of gradients
            loss.backward()

            # Update mask
            optimizer.step()

            with torch.no_grad():
                mask_add.copy_(projection(mask_add))

            mask_add_bool = self.random_sampling(mask_add.clone())
            mask_remove_bool = mask_remove.bool()
            mask_joint_bool = torch.cat([mask_remove_bool, mask_add_bool], dim=0)

            edge_index_joint_filtered = edge_index_joint[:, mask_joint_bool]
            out = model(data.x, edge_index_joint_filtered)
            train_acc, test_acc, train_stats, test_stats = metrics(out, data)
            print(f"epoch: {t}, loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")


    def random_sampling(self, mask):
        random_matrix = torch.rand_like(mask)
        u = random_matrix < mask
        return u

    def generate_edge_index_can_add(self, edge_index, N):
        # Создаём все возможные рёбра с использованием torch.cartesian_prod
        all_edges = torch.cartesian_prod(torch.arange(N), torch.arange(N)).t()

        # Преобразуем edge_index в множество для быстрого поиска
        edge_index_set = set(map(tuple, edge_index.t().tolist()))

        # Преобразуем all_edges в множество для вычитания
        all_edges_set = set(map(tuple, all_edges.t().tolist()))

        # Вычитаем рёбра из edge_index
        remaining_edges = all_edges_set - edge_index_set

        # Преобразуем оставшиеся рёбра обратно в тензор формата (2, E)
        edge_index_add = torch.tensor(list(remaining_edges)).t()

        return edge_index_add


    @staticmethod
    # TODO функция attack_loss должна совпадать с фунцкией потерь, используемой в процессе обучения модели
    #  (предложение авторов статьи). Поэтому следует расширить функционал атаки для различных loss функций
    def __attack_loss(preds, labels):
        return F.cross_entropy(preds, labels)


class NettackEvasionAttacker(EvasionAttacker):
    name = "NettackEvasionAttacker"

    def __init__(self,
                 node_idx=0,
                 n_perturbations=None,
                 perturb_features=True,
                 perturb_structure=True,
                 direct=True,
                 n_influencers=0
                 ):

        super().__init__()
        self.attack_diff = None
        self.node_idx = node_idx
        self.n_perturbations = n_perturbations
        self.perturb_features = perturb_features
        self.perturb_structure = perturb_structure
        self.direct = direct
        self.n_influencers = n_influencers

    def attack(self, model_manager, gen_dataset, mask_tensor):
        # Prepare
        data = gen_dataset.data
        _A_obs, _X_obs, _z_obs = data_to_csr_matrix(data)
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        lcc = largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:, lcc]

        assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
        assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        _X_obs = _X_obs[lcc].astype('float32')
        _z_obs = _z_obs[lcc]
        _N = _A_obs.shape[0]
        _K = _z_obs.max() + 1
        _Z_obs = np.eye(_K)[_z_obs]
        _An = preprocess_graph(_A_obs)
        degrees = _A_obs.sum(0).A1

        if self.n_perturbations is None:
            self.n_perturbations = int(degrees[self.node_idx])
        hidden = model_manager.gnn.GCNConv_0.out_channels
        # End prepare

        # Learn matrix W1 and W2
        W1, W2 = train_w1_w2(dataset=gen_dataset, hidden=hidden)

        # Attack
        nettack = Nettack(_A_obs, _X_obs, _z_obs, W1, W2, self.node_idx, verbose=True)

        nettack.reset()
        nettack.attack_surrogate(n_perturbations=self.n_perturbations,
                                 perturb_structure=self.perturb_structure,
                                 perturb_features=self.perturb_features,
                                 direct=self.direct,
                                 n_influencers=self.n_influencers)

        print(f'edges: {nettack.structure_perturbations}')
        print(f'features: {nettack.feature_perturbations}')

        self._evasion(gen_dataset, nettack.feature_perturbations, nettack.structure_perturbations)
        self.attack_diff = gen_dataset

        return gen_dataset

    def attack_diff(self):
        return self.attack_diff

    @staticmethod
    def _evasion(gen_dataset, feature_perturbations, structure_perturbations):
        cleaned_feat_pert = list(filter(None, feature_perturbations))
        if cleaned_feat_pert:  # list is not empty
            x = gen_dataset.data.x.clone()
            for vertex, feature in cleaned_feat_pert:
                if x[vertex, feature] == 0.0:
                    x[vertex, feature] = 1.0
                elif x[vertex, feature] == 1.0:
                    x[vertex, feature] = 0.0
            gen_dataset.data.x = x

        cleaned_struct_pert = list(filter(None, structure_perturbations))
        if cleaned_struct_pert:  # list is not empty
            edge_index = gen_dataset.data.edge_index.clone()
            # add edges
            for edge in cleaned_struct_pert:
                edge_index = torch.cat((edge_index,
                                        torch.tensor((edge[0], edge[1]), dtype=torch.int32).to(torch.int64).unsqueeze(1)), dim=1)
                edge_index = torch.cat((edge_index,
                                        torch.tensor((edge[1], edge[0]), dtype=torch.int32).to(torch.int64).unsqueeze(1)), dim=1)

            gen_dataset.data.edge_index = edge_index

class NettackGroupEvasionAttacker(EvasionAttacker):
    name = "NettackGroupEvasionAttacker"
    def __init__(self,node_idxs, **kwargs):
        super().__init__()
        self.node_idxs = node_idxs # kwargs.get("node_idxs")
        assert isinstance(self.node_idxs, list)
        self.n_perturbations = kwargs.get("n_perturbations")
        self.perturb_features = kwargs.get("perturb_features")
        self.perturb_structure = kwargs.get("perturb_structure")
        self.direct = kwargs.get("direct")
        self.n_influencers = kwargs.get("n_influencers")
        self.attacker = NettackEvasionAttacker(0, **kwargs)

    def attack(self, model_manager, gen_dataset, mask_tensor):
        for node_idx in self.node_idxs:
            self.attacker.node_idx = node_idx
            gen_dataset = self.attacker.attack(model_manager, gen_dataset, mask_tensor)
        return gen_dataset