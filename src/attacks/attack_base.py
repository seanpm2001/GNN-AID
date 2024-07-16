import numpy as np
import torch

from base.datasets_processing import GeneralDataset


class Attacker:
    name = "Attacker"

    def __init__(self, gen_dataset: GeneralDataset, model):
        self.gen_dataset = gen_dataset
        self.model = model

    def attack(self):
        pass

    def save(self, path):
        pass

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        return False


class RandomPoisonAttack(Attacker):
    def __init__(self, gen_dataset: GeneralDataset, model, n_edges_percent=0.1):
        super().__init__(gen_dataset=gen_dataset, model=model)
        self.n_edges_percent = n_edges_percent

    def attack(self):
        edge_index = self.gen_dataset.data.edge_index
        random_indices = np.random.choice(
            edge_index.shape[1],
            int(edge_index.shape[1] * (1 - self.n_edges_percent)),
            replace=False
        )
        random_indices = np.sort(random_indices)
        edge_index = edge_index[:, random_indices]
        self.gen_dataset.data.edge_index = edge_index
