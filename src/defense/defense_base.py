import numpy as np

from aux.configs import PoisonDefenseConfig, MIDefenseConfig, EvasionDefenseConfig, ConfigPattern, PoisonAttackConfig, \
    CONFIG_OBJ
from base.datasets_processing import GeneralDataset


class Defender:
    name = "Defender"

    def __init__(self):
        pass

    def defense(self, **kwargs):
        pass

    def defense_diff(self):
        pass

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        return False


class EvasionDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()


class MIDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()


class PoisonDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()


class BadRandomPoisonDefender(PoisonDefender):
    name = "BadRandomPoisonDefender"

    def __init__(self, n_edges_percent=0.1):
        self.defense_diff = None

        super().__init__()
        self.n_edges_percent = n_edges_percent

    def defense(self, gen_dataset):
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
        self.defense_diff = edge_index_diff
        return gen_dataset

    def defense_diff(self):
        return self.defense_diff
