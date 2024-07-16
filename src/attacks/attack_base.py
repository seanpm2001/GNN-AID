import numpy as np
import torch

from aux.configs import PoisonAttackConfig, EvasionAttackConfig, MIAttackConfig, ConfigPattern, CONFIG_OBJ
from base.datasets_processing import GeneralDataset


class Attacker:
    name = "Attacker"

    def __init__(self, gen_dataset: GeneralDataset, model):
        self.gen_dataset = gen_dataset
        self.model = model

    def attack(self):
        pass

    def attack_diff(self):
        pass

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        return False


class EvasionAttacker(Attacker):
    def __init__(self, gen_dataset: GeneralDataset, model,
                 evasion_attack_config=None):
        super().__init__(gen_dataset, model)
        self.evasion_attack_config = evasion_attack_config


class MIAttacker(Attacker):
    def __init__(self, gen_dataset: GeneralDataset, model,
                 mi_attack_config=None):
        super().__init__(gen_dataset, model)
        self.mi_attack_config = mi_attack_config


class PoisonAttacker(Attacker):
    def __init__(self, gen_dataset: GeneralDataset, model,
                 poison_attack_config=None):
        super().__init__(gen_dataset, model)
        self.poison_attack_config = poison_attack_config


class RandomPoisonAttack(PoisonAttacker):
    name = "RandomPoisonAttack"
    def __init__(self, gen_dataset: GeneralDataset, model, poison_attack_config):
        self.attack_diff = None

        if poison_attack_config is None:
            poison_attack_config = ConfigPattern(
                _config_class="PoisonAttackConfig",
                _config_kwargs={},
            )
        elif isinstance(poison_attack_config, PoisonAttackConfig):
            poison_attack_config = ConfigPattern(
                _config_class="PoisonAttackConfig",
                _config_kwargs=poison_attack_config.to_dict(),
            )

        super().__init__(gen_dataset, model, poison_attack_config)
        self.n_edges_percent = getattr(self.poison_attack_config, CONFIG_OBJ).n_edges_percent

    def attack(self):
        edge_index = self.gen_dataset.data.edge_index
        random_indices = np.random.choice(
            edge_index.shape[1],
            int(edge_index.shape[1] * (1 - self.n_edges_percent)),
            replace=False
        )
        total_indices_array = np.arange(edge_index.shape[1])
        indices_to_remove = np.setdiff1d(total_indices_array, random_indices)
        edge_index_diff = edge_index[:, indices_to_remove]
        edge_index = edge_index[:, random_indices]
        self.gen_dataset.data.edge_index = edge_index
        self.attack_diff = edge_index_diff

    def attack_diff(self):
        return self.attack_diff
