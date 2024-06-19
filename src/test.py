import torch

import warnings

from torch import device

from aux.utils import OPTIMIZERS_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_INIT_PARAMETERS_PATH
# from explainers.explainers_manager import FrameworkExplainersManager
# from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from aux.configs import ModelManagerConfig, ModelModificationConfig, ExplainerInitConfig, ExplainerRunConfig, \
    ConfigPattern, ExplainerModificationConfig
from base.datasets_processing import DatasetManager
from models_builder.models_zoo import model_configs_zoo


def test_SubgraphX():
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

    # gcn2 = GNNStructure(
    #     # conv_classes=('SGConv', 'SGConv'),
    #     conv_classes=('GCNConv', 'GCNConv'),
    #     layers_sizes=(dataset.num_node_features, 16, dataset.num_classes),
    #     activations=('torch.relu', 'torch.nn.functional.log_softmax'),
    #     conv_kwargs={}
    # )

    # # /home/lukyanovkirill/Projects/MUTAG.pkl
    # class MyGNNModelManager(GNNModelManager):
    #     def __init__(self, dataset_path):
    #         super().__init__()
    #         self.dataset_path = dataset_path
    #
    # mm = MyGNNModelManager(dataset_path=results_dataset_path)
    # mm.load_model(path='/home/lukyanovkirill/Projects/MUTAG')
    # # mm.load_model(path='/home/lukyanovkirill/Projects/model')
    # print(mm.gnn)

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='test_gnn')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin')
    # gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin_prot')

    # try:
    #     print(gnn.forward())
    # except NotImplementedError:
    #     print('NotImplementedError')
    # except:
    #     print('Error, but function implement and callable')

    # manager_config = ConfigPattern(
    #     _config_class="ModelManagerConfig",
    #     _config_kwargs={
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
    #
    # # train_test_split = [0.8, 0.2]
    # # train_test_split = [0.6, 0.4]
    # steps_epochs = 200
    # gnn_model_manager = FrameworkGNNModelManager(
    #     gnn=gnn,
    #     dataset_path=results_dataset_path,
    #     manager_config=manager_config,
    #     modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    # )
    #
    # # save_model_flag = False
    # save_model_flag = True
    #
    # # data.x = data.x.float()
    # gnn_model_manager.gnn.to(my_device)
    # data = data.to(my_device)
    #
    # warnings.warn("Start training")
    # dataset.train_test_split()
    #
    # try:
    #     # raise FileNotFoundError()
    #     gnn_model_manager.load_model_executor()
    # except FileNotFoundError:
    #     gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
    #     train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
    #                                                           save_model_flag=save_model_flag,
    #                                                           metrics=[Metric("F1", mask='train', average=None)])
    #
    #     if train_test_split_path is not None:
    #         dataset.save_train_test_mask(train_test_split_path)
    #         train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
    #                                                             :]
    #         dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
    #         data.percent_train_class, data.percent_test_class = train_test_sizes
    #
    # warnings.warn("Training was successful")
    #
    # metric_loc = gnn_model_manager.evaluate_model(
    #     gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    # print(metric_loc)

    # embeddings = gnn_model_manager.gnn.get_all_layer_embeddings(
    #     dataset.dataset._data.x, dataset.dataset._data.edge_index
    # )
    # print(embeddings[0].tolist()[1])
    # f_list = []
    # for j in [0, 633, 926, 1166, 1701, 1862, 1866, 2582]:
    #     f_list += (set(i for i, elem in enumerate(data.x[j].tolist()) if elem > 0))
    # f_list_sort = sorted(f_list)
    #
    # f_dict = {}
    # for elem in f_list_sort:
    #     if elem in f_dict:
    #         f_dict[elem] += 1
    #     else:
    #         f_dict[elem] = 1
    # print(f_dict)
    # print(list((key, val) for key, val in f_dict.items() if val > 1))

    # print(list(i for i, elem in enumerate(data.x[2582].tolist()) if elem > 0))

    # 0    [19, 81,  146, 315, 774, 877, 1194, 1247, 1274]
    # 633  [19, 52,  214, 226, 353, 357, 494,  548,  621,  720,  723,  774,  1075, 1209, 1251, 1301, 1381, 1389, 1392]
    # 926  [19, 48,  177, 540, 615, 737, 742,  774,  908,  950,  969,  1076, 1105, 1218, 1301, 1355]
    # 1166 [19, 27,  48,  55,  93,  130, 464,  510,  667,  723,  763,  774,  923,  959,  1123, 1141, 1198, 1209, 1219, 1328, 1347, 1363, 1389, 1392]
    # 1701 [85, 132, 445, 540, 543, 558, 588,  973,  1060, 1076, 1138, 1263, 1272, 1280, 1295, 1299, 1353, 1357, 1361, 1384]
    # 1862 [19, 41,  305, 510, 540, 647, 720,  774,  855,  1075, 1156, 1308, 1389, 1392, 1431]
    # 1866 [19, 48,  52,  156, 353, 548, 774,  1146, 1177, 1198, 1209, 1249, 1266, 1301, 1330, 1366, 1392, 1426]
    # 2582 [19, 98,  316, 360, 393, 469, 548,  860,  1075, 1097, 1123, 1132, 1144, 1148, 1202, 1266, 1305, 1308, 1418]


    # Explain node 10
    node = 0

    # Explanation size
    max_nodes = 5

    # warnings.warn("Start SubgraphX")
    # explainer_init_config = ConfigPattern(
    #     _class_name="SubgraphX",
    #     _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
    #     _config_class="ExplainerInitConfig",
    #     _config_kwargs={
    #         # "class_name": "SubgraphX",
    #     }
    # )
    # explainer_run_config = ConfigPattern(
    #     _config_class="ExplainerRunConfig",
    #     _config_kwargs={
    #         "mode": "local",
    #         "kwargs": {
    #             "_class_name": "GNNExplainer(torch-geom)",
    #             "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    #             "_config_class": "Config",
    #             "_config_kwargs": {
    #                 "element_idx": 0, "max_nodes": 5
    #             },
    #         }
    #     }
    # )
    # explainer_SubgraphX = FrameworkExplainersManager(
    #     init_config=explainer_init_config,
    #     dataset=dataset, gnn_manager=gnn_model_manager,
    #     explainer_name="SubgraphX",
    # )
    # explainer_SubgraphX.conduct_experiment(explainer_run_config)
    #
    #
    # return metric_loc


if __name__ == '__main__':
    test_SubgraphX()

