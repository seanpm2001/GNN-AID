import numpy as np
import importlib
import torch

from attacks.attack_base import Attacker
from pathlib import Path

# PGD imports
from attacks.poison_attacks_collection.pgd.utils import Projection, RandomSampling
from attacks.poison_attacks_collection.pgd.model import MatGNN
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

POISON_ATTACKS_DIR = Path(__file__).parent.resolve() / 'poison_attacks_collection'

class PoisonAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class EmptyPoisonAttacker(PoisonAttacker):
    name = "EmptyPoisonAttacker"

    def attack(self, **kwargs):
        pass


class RandomPoisonAttack(PoisonAttacker):
    name = "RandomPoisonAttack"

    def __init__(self, n_edges_percent=0.1):
        self.attack_diff = None

        super().__init__()
        self.n_edges_percent = n_edges_percent

    def attack(self, gen_dataset):
        edge_index = gen_dataset.data.edge_index
        random_indices = np.random.choice(
            edge_index.shape[1],
            int(edge_index.shape[1] * (1 - self.n_edges_percent)),
            replace=False
        )
        total_indices_array = np.arange(edge_index.shape[1])
        indices_to_remove = np.setdiff1d(total_indices_array, random_indices)
        edge_index_diff = edge_index[:, indices_to_remove]
        edge_index = edge_index[:, random_indices]
        gen_dataset.data.edge_index = edge_index
        self.attack_diff = edge_index_diff
        return gen_dataset

    def attack_diff(self):
        return self.attack_diff

class EmptyPoisonAttacker(PoisonAttacker):
    name = "EmptyPoisonAttacker"

    def attack(self, **kwargs):
        pass

# for attack_name in POISON_ATTACKS_DIR.rglob("*_attack.py"):
#     try:
#         importlib.import_module(str(attack_name))
#     except ImportError:
#         print(f"Couldn't import Attack: {attack_name}")

# import attacks.poison_attacks_collection.metattack.meta_gradient_attack

# # TODO this is not best practice to import this thing here this way
# from attacks.poison_attacks_collection.metattack.meta_gradient_attack import BaseMeta

class PGDAttacker(PoisonAttacker):
    name = "PGD"

    def __init__(self,
                 perturb_ratio=0.5,
                 learning_rate=0.01,
                 num_iterations=100,
                 num_rand_trials=100):

        super().__init__()
        self.attack_diff = None
        self.perturb_ratio = perturb_ratio
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_rand_trials = num_rand_trials

    def attack(self, model_manager, gen_dataset):
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

    @staticmethod
    # TODO функция attack_loss должна совпадать с фунцкией потерь, используемой в процессе обучения модели
    #  (предложение авторов статьи). Поэтому следует расширить функционал атаки для различных loss функций
    def __attack_loss(preds, labels):
        return -F.cross_entropy(preds, labels)

