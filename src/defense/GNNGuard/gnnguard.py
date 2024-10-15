import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv

# from defense.GNNGuard.base_model import BaseModel
from defense.poison_defense import PoisonDefender

from models_builder.gnn_models import FrameworkGNNModelManager
from models_builder.gnn_constructor import FrameworkGNNConstructor
from models_builder.models_zoo import model_configs_zoo
from aux.configs import ModelManagerConfig, ModelModificationConfig, DatasetConfig, DatasetVarConfig, ConfigPattern
from aux.utils import import_by_name, CUSTOM_LAYERS_INFO_PATH, MODULES_PARAMETERS_PATH, hash_data_sha256, \
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY, OPTIMIZERS_PARAMETERS_PATH


import warnings
import types
# from torch_sparse import coalesce, SparseTensor, matmul

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np

from src.aux.configs import ModelConfig


class BaseGNNGuard(PoisonDefender):
    name = "BaseGNNGuard"

    def __init__(self, lr=0.1, train_iters=200, device='cpu'):
        super().__init__()
        self.model = None
        self.lr = lr
        self.device = device
        self.train_iters = train_iters
    
    def defense(self, gen_dataset):
        self.model = model_configs_zoo(gen_dataset, 'gcn_gcn_linearized')
        default_config = ModelModificationConfig(
            model_ver_ind=0,
        )
        manager_config = ConfigPattern(
            _config_class="ModelManagerConfig",
            _config_kwargs={
                "mask_features": [],
                "optimizer": {
                    "_config_class": "Config",
                    "_class_name": "Adam",
                    "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                    "_class_import_info": ["torch.optim"],
                    "_config_kwargs": {"weight_decay": 5e-4},
                }
            }
        )
        gnn_model_manager_surrogate = FrameworkGNNModelManager(
            gnn=self.model,
            dataset_path=gen_dataset,
            modification=default_config,
            manager_config=manager_config,
        )

        gnn_model_manager_surrogate.train_model(gen_dataset=gen_dataset, steps=self.train_iters)

        self.pred_labels = gnn_model_manager_surrogate.run_model(gen_dataset=gen_dataset, mask='all', out='answers')


class GNNGuard(BaseGNNGuard):
    name = 'GNNGuard'

    def __init__(self, lr=0.01, attention=False, drop=False, train_iters=200, device='cpu', with_bias=False, with_relu=False):
        super().__init__(lr=lr, train_iters=train_iters, device=device)
        assert device is not None, "Please specify 'device'!"
        self.with_bias = with_bias
        self.with_relu = with_relu
        self.attention = attention
        self.drop = drop

    def defense(self, gen_dataset):
        super().defense(gen_dataset=gen_dataset)
        
        self.hidden_sizes = [16]
        self.nfeat = gen_dataset.num_node_features
        self.nclass = gen_dataset.num_classes
        # self.attention = attention
        # self.drop = drop

        self.wrap()
        self.initialize()

    def wrap(self):
        def att_coef(self, fea, edge_index, is_lil=False, i=0):
            if is_lil == False:
                edge_index = edge_index._indices()
            else:
                edge_index = edge_index.tocoo()

            n_node = fea.shape[0]
            row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

            fea_copy = fea.cpu().data.numpy()
            sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
            sim = sim_matrix[row, col]
            sim[sim<0.1] = 0
            
            """build a attention matrix"""
            att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
            att_dense[row, col] = sim
            if att_dense[0, 0] == 1:
                att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
            # normalization, make the sum of each row is 1
            att_dense_norm = normalize(att_dense, axis=1, norm='l1')


            """add learnable dropout, make character vector"""
            if self.drop:
                character = np.vstack((att_dense_norm[row, col].A1,
                                        att_dense_norm[col, row].A1))
                character = torch.from_numpy(character.T)
                drop_score = self.drop_learn_1(character)
                drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
                mm = torch.nn.Threshold(0.5, 0)
                drop_score = mm(drop_score)
                mm_2 = torch.nn.Threshold(-0.49, 1)
                drop_score = mm_2(-drop_score)
                drop_decision = drop_score.clone().requires_grad_()
                # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
                drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
                drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
                att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

            if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
                degree = (att_dense_norm != 0).sum(1).A1
                lam = 1 / (degree + 1) # degree +1 is to add itself
                self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
                att = att_dense_norm + self_weight  # add the self loop
            else:
                att = att_dense_norm

            row, col = att.nonzero()
            att_adj = np.vstack((row, col))
            att_edge_weight = att[row, col]
            att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
            att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)#.cuda()
            att_adj = torch.tensor(att_adj, dtype=torch.int64)#.cuda()

            shape = (n_node, n_node)
            new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
            return new_adj
        
        def forward(self, *args, **kwargs):
            """we don't change the edge_index, just update the edge_weight;
            some edge_weight are regarded as removed if it equals to zero"""
            layer_ind = -1
            tensor_storage = {}
            dim_cat = 0
            layer_emb_dict = {}
            save_emb_flag = self._save_emb_flag

            x, edge_index, batch = self.arguments_read(*args, **kwargs)
            feat = x
            adj = edge_index.tocoo()
            adj_memory = None
            # print(list(self.__dict__['_modules'].items()))
            for elem in list(self.__dict__['_modules'].items()):
                layer_name, curr_layer_ind = elem[0].split('_')
                curr_layer_ind = int(curr_layer_ind)
                inp = torch.clone(x)
                loc_flag = False
                if curr_layer_ind != layer_ind:
                    if save_emb_flag:
                        loc_flag = True
                    zeroing_x_flag = False
                    for key, value in self.conn_dict.items():
                        if key[0] == layer_ind and layer_ind not in tensor_storage:
                            tensor_storage[layer_ind] = torch.clone(x)
                    layer_ind = curr_layer_ind
                    x_copy = torch.clone(x)
                    connection_tensor = torch.Tensor()
                    for key, value in self.conn_dict.items():

                        if key[1] == curr_layer_ind:
                            if key[1] - key[0] == 1:
                                zeroing_x_flag = True
                            for con in self.conn_dict[key]:
                                if self.embedding_levels_by_layers[key[1]] == 'n' and \
                                        self.embedding_levels_by_layers[key[0]] == 'n':
                                    connection_tensor = torch.cat((connection_tensor,
                                                                tensor_storage[key[0]]), 1)
                                    dim_cat = 1
                                elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                        self.embedding_levels_by_layers[key[0]] == 'g':
                                    connection_tensor = torch.cat((connection_tensor,
                                                                tensor_storage[key[0]]), 0)
                                    dim_cat = 0
                                elif self.embedding_levels_by_layers[key[1]] == 'g' and \
                                        self.embedding_levels_by_layers[key[0]] == 'n':
                                    con_pool = import_by_name(con['pool']['pool_type'],
                                                            ["torch_geometric.nn"])
                                    tensor_after_pool = con_pool(tensor_storage[key[0]], batch)
                                    connection_tensor = torch.cat((connection_tensor,
                                                                tensor_after_pool), 1)
                                    dim_cat = 1
                                else:
                                    raise Exception(
                                        "Connection from layer type "
                                        f"{self.embedding_levels_by_layers[curr_layer_ind - 1]} to"
                                        f" layer type {self.embedding_levels_by_layers[curr_layer_ind]}"
                                        "is not supported now")
                    

                    if zeroing_x_flag:
                        x = connection_tensor
                    else:
                        x = torch.cat((x_copy, connection_tensor), dim_cat)
                    

                    if self.attention:
                        if layer_name == 'GINConv':
                            if adj_memory is None:
                                adj = self.att_coef(x, adj, is_lil=False,i=layer_ind)
                                edge_index = adj._indices()
                                edge_weight = adj._values()
                                adj_memory = adj
                            elif adj_memory is not None:
                                adj = self.att_coef(x, adj_memory, is_lil=False, i=layer_ind)
                                edge_weight = self.gate * adj_memory._values() + (1 - self.gate) * adj._values()
                                adj_memory = adj
                        elif layer_name == 'GCNConv' or layer_name == 'GATConv':
                            if adj_memory is None:
                                adj = self.att_coef(x, adj, i=0)
                                edge_index = adj._indices()
                                edge_weight = adj._values()
                                adj_memory = adj
                            elif adj_memory is not None:
                                adj = self.att_coef(x, adj_memory, i=layer_ind).to_dense()
                                row, col = adj.nonzero()[:,0], adj.nonzero()[:,1]
                                edge_index = torch.stack((row, col), dim=0)
                                edge_weight = adj[row, col]
                                adj_memory = adj
                    else:
                        edge_index = adj._indices()
                        edge_weight = adj._values()


                # QUE Kirill, maybe we should not off UserWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    # mid = x
                    if layer_name in self.modules_info:
                        code_str = f"getattr(self, elem[0])" \
                                    f"({self.modules_info[layer_name][TECHNICAL_PARAMETER_KEY]['forward_parameters']}," \
                                    f" edge_weight=edge_weight)"
                        x = eval(f"{code_str}")
                    else:
                        x = getattr(self, elem[0])(x)
                if loc_flag:
                    layer_emb_dict[layer_ind] = torch.clone(x)

            if save_emb_flag:
                return layer_emb_dict
            return x
        
        self.model.drop = self.drop
        self.model.attention = self.attention
        self.model.gate = Parameter(torch.rand(1))
        self.model.drop_learn_1 = nn.Linear(2, 1)
        self.model.drop_learn_2 = nn.Linear(2, 1)
        self.model.forward = types.MethodType(forward, self.model)
        self.model.att_coef = types.MethodType(att_coef, self.model)

    def initialize(self):
        self.model.drop_learn_1.reset_parameters()
        self.model.drop_learn_2.reset_parameters()
        # self.model.gate.reset_parameters()
