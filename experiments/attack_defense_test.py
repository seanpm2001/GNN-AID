import torch

import warnings

from torch import device

from attacks.attack_base import RandomPoisonAttack
from defense.defense_base import BadRandomPoisonDefender
from models_builder.attack_defense_manager import AttackAndDefenseManager
from src.aux.utils import OPTIMIZERS_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_INIT_PARAMETERS_PATH, POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH
from src.explainers.explainers_manager import FrameworkExplainersManager
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelManagerConfig, ModelModificationConfig, ExplainerInitConfig, ExplainerRunConfig, \
    ConfigPattern, ExplainerModificationConfig
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo


def test_attack_defense():
    # my_device = device('cuda' if is_available() else 'cpu')
    my_device = device('cpu')

    full_name = None

    # full_name = ("multiple-graphs", "TUDataset", 'MUTAG')
    # full_name = ("single-graph", "custom", 'karate')
    full_name = ("single-graph", "Planetoid", 'Cora')
    # full_name = ("multiple-graphs", "TUDataset", 'PROTEINS')

    # dataset, data, results_dataset_path = DatasetManager.get_pytorch_geometric(
    #     full_name=("single-graph", "Planetoid", 'Cora'),
    #     dataset_attack_type='original')
    # dataset, data, results_dataset_path = DatasetManager.get_pytorch_geometric(
    #     full_name=("single-graph", "pytorch-geometric-other", 'KarateClub'),
    #     dataset_attack_type='original',
    #     dataset_ver_ind=0)

    # dataset, data, results_dataset_path = DatasetManager.get_pytorch_geometric(
    #     full_name=("single-graph", "Planetoid", 'Cora'),
    #     dataset_attack_type='original',
    #     dataset_ver_ind=0)
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_attack_type='original',
        dataset_ver_ind=0
    )

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     full_name=("single-graph", "custom", "example",),
    #     features={'attr': {'a': 'as_is', 'b': 'as_is'}},
    #     dataset_attack_type='original',
    #     labeling='threeClasses',
    #     dataset_ver_ind=0
    # )

    # dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    #     # full_name=("single-graph", "vk_samples", "vk2-ff40-N100000-A.1612175945",),
    #     full_name=("single-graph", "vk_samples", "vk2-ff20-N10000-A.1611943634",),
    #     # full_name=("single-graph", "vk_samples", "vk2-ff20-N1000-U.1612273925",),
    #     # features=('sex',),
    #     features={'str_f': tuple(), 'str_g': None, 'attr': {
    #         # "('personal', 'political')": 'one_hot',
    #         # "('occupation', 'type')": 'one_hot', # Don't work now
    #         # "('relation',)": 'one_hot',
    #         # "('age',)": 'one_hot',
    #         "('sex',)": 'one_hot',
    #     }},
    #     # features={'str_f': tuple(), 'str_g': None, 'attr': {'sex': 'one_hot', }},
    #     labeling='sex1',
    #     dataset_attack_type='original',
    #     dataset_ver_ind=0
    # )

    # print(data.train_mask)

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='test_gnn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin_prot')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            }
        }
    )
    # manager_config = ModelManagerConfig(**{
    #         "mask_features": [],
    #         "optimizer": {
    #             # "_config_class": "Config",
    #             "_class_name": "Adam",
    #             # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
    #             # "_class_import_info": ["torch.optim"],
    #             "_config_kwargs": {},
    #         }
    #     }
    # )

    # train_test_split = [0.8, 0.2]
    # train_test_split = [0.6, 0.4]
    steps_epochs = 200
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )

    save_model_flag = False
    # save_model_flag = True

    # data.x = data.x.float()
    gnn_model_manager.gnn.to(my_device)
    data = data.to(my_device)

    poison_attack_config = ConfigPattern(
        _class_name="RandomPoisonAttack",
        _import_path=POISON_ATTACK_PARAMETERS_PATH,
        _config_class="PoisonAttackConfig",
        _config_kwargs={
            "n_edges_percent": 0.1,
        }
    )

    poison_defense_config = ConfigPattern(
        _class_name="BadRandomPoisonDefender",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
            "n_edges_percent": 0.1,
        }
    )

    # random_attack = RandomPoisonAttack(gen_dataset=dataset, model=gnn, poison_attack_config=poison_attack_config)
    # random_attack.attack()
    #
    # bad_defense = BadRandomPoisonDefender(gen_dataset=dataset, model=gnn, poison_defense_config=poison_defense_config)
    # bad_defense.defense()

    attack_defense_manager = AttackAndDefenseManager(gen_dataset=dataset, gnn_manager=gnn_model_manager)
    attack_defense_manager.set_poison_attacker(poison_attack_config=poison_attack_config)
    attack_defense_manager.set_poison_defender(poison_defense_config=poison_defense_config)

    attack_defense_manager.conduct_experiment()

    warnings.warn("Start training")
    dataset.train_test_split()

    try:
        raise FileNotFoundError()
        # gnn_model_manager.load_model_executor()
    except FileNotFoundError:
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes

    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)



if __name__ == '__main__':
    test_attack_defense()


