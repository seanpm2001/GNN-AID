"""
Experiment on attacking GNN via GNNExplainer's explanations
"""

import torch
import numpy as np
import warnings

from torch import device

from src.aux.utils import POISON_ATTACK_PARAMETERS_PATH, POISON_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, \
    EVASION_DEFENSE_PARAMETERS_PATH
from src.models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern, CONFIG_OBJ
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo

from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager

from explainers.GNNExplainer.torch_geom_our.out import GNNExplainer


def test():
    from attacks.EGNNE.egnne_attack import EAttack

    my_device = device('cpu')

    # Load dataset
    full_name = ("single-graph", "Planetoid", 'Cora')
    # full_name = ('single-graph', 'pytorch-geometric-other', 'KarateClub')
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # Train model on original dataset and remember the model metric and node predicted probability
    gcn_gcn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')

    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                "_class_name": "Adam",
                "_config_kwargs": {},
            }
        }
    )

    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gcn_gcn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=0)
    )

    gnn_model_manager.gnn.to(my_device)

    num_steps = 100
    gnn_model_manager.train_model(gen_dataset=dataset,
                                  steps=num_steps,
                                  save_model_flag=False)

    # Evaluate model

    acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                 metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']


    acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    print(f"BEFORE ATTACK\nAccuracy on train: {acc_train}. Accuracy on test: {acc_test}")
    # print(f"Accuracy on test: {acc_test}")

    explainer_init_config = ConfigPattern(
        _class_name="GNNExplainer(torch-geom)",
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={
        }
    )
    explainer_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": "GNNExplainer(torch-geom)",
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": {

                },
            }
        }
    )

    # explainer_GNNExpl = FrameworkExplainersManager(
    #     init_config=explainer_init_config,
    #     dataset=dataset, gnn_manager=gnn_model_manager,
    #     explainer_name='GNNExplainer(torch-geom)',
    # )

    init_kwargs = getattr(explainer_init_config, CONFIG_OBJ).to_dict()
    explainer = GNNExplainer(gen_dataset=dataset, model=gnn_model_manager.gnn, device=device, **init_kwargs)

    node_inds = np.arange(dataset.dataset.data.x.shape[0])
    # dataset = gen_dataset.dataset.data[mask_tensor]
    # num_nodes = len(node_inds)
    # attacked_node_size = int(num_nodes * self.attack_size)
    attacked_node_size = int((0.15 * len(node_inds)))
    attack_inds = np.random.choice(node_inds, attacked_node_size)

    evasion_attack_config = ConfigPattern(
        _class_name="EAttack",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            'explainer': explainer,
            'run_config': explainer_run_config,
            'mode': 'local',
            'attack_inds': attack_inds
        }
    )

    gnn_model_manager.set_evasion_attacker(evasion_attack_config=evasion_attack_config)

    mask = Metric.create_mask_by_target_list(y_true=dataset.labels, target_list=attack_inds)

    # explainer_GNNExpl.conduct_experiment(explainer_run_config)

    # Evaluate model

    acc_attack = gnn_model_manager.evaluate_model(gen_dataset=dataset,
                                                 metrics=[Metric("Accuracy", mask=mask)])[mask]['Accuracy']
    print(f"AFTER ATTACK\nAccuracy: {acc_attack}")

    # acc_train = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                              metrics=[Metric("Accuracy", mask='train')])['train']['Accuracy']
    #
    #
    # acc_test = gnn_model_manager.evaluate_model(gen_dataset=dataset,
    #                                             metrics=[Metric("Accuracy", mask='test')])['test']['Accuracy']
    # print(f"AFTER ATTACK\nAccuracy on train: {acc_train}. Accuracy on test: {acc_test}")






if __name__ == "__main__":
    test()