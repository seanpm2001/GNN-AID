import numpy as np
import torch

from attacks.attack_base import Attacker

# Nettack imports
from src.attacks.nettack.nettack import Nettack
from src.attacks.nettack.utils import preprocess_graph, largest_connected_components, data_to_csr_matrix, learn_w1_w2
from torch_geometric.data import Data
# Nettack imports end


class PoisonAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class NettackPoisonAttack(PoisonAttacker):
    name = "NettackPoisonAttack"

    def __init__(self,
                 node_idx=0,
                 direct_attack=True,
                 n_influencers=5,
                 perturb_features=True,
                 perturb_structure=True):

        super().__init__()
        self.attack_diff = None
        self.node_idx = node_idx
        self.direct_attack = direct_attack
        self.n_influencers = n_influencers
        self.perturb_features = perturb_features
        self.perturb_structure = perturb_structure

    def attack(self, gen_dataset):
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
        sizes = [16, _K]
        degrees = _A_obs.sum(0).A1
        n_perturbations = int(degrees[self.node_idx])
        # n_perturbations = 3
        # End prepare

        # Learn matrix W1 and W2
        W1, W2 = learn_w1_w2(gen_dataset)

        # Attack
        nettack = Nettack(_A_obs, _X_obs, _z_obs, W1, W2, self.node_idx, verbose=True)

        nettack.reset()
        nettack.attack_surrogate(n_perturbations,
                                 perturb_structure=self.perturb_structure,
                                 perturb_features=self.perturb_features,
                                 direct=self.direct_attack,
                                 n_influencers=self.n_influencers)

        print(f'edges: {nettack.structure_perturbations}')
        print(f'features: {nettack.feature_perturbations}')

        self._poisoning(gen_dataset, nettack.feature_perturbations, nettack.structure_perturbations)
        self.attack_diff = gen_dataset

        return gen_dataset

    def attack_diff(self):
        return self.attack_diff

    @staticmethod
    def _poisoning(gen_dataset, feature_perturbations, structure_perturbations):
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
