import numpy as np
import random
import torch
import copy

from explainers.GNNExplainer.torch_geom_our.out import GNNExplainer
from explainers.SubgraphX.out import SubgraphXExplainer
from explainers.Zorro.out import ZorroExplainer
from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, \
    EXPLAINERS_GLOBAL_RUN_PARAMETERS_PATH
from aux.configs import ConfigPattern

from tqdm import tqdm
from networkx.classes import neighbors
from numpy.array_api import astype
from sympy.codegen.ast import int64

from attacks.evasion_attacks import EvasionAttacker
from aux.configs import CONFIG_OBJ
from explainers.explainer import ProgressBar
from typing import Dict, Optional


class EAttack(EvasionAttacker):
    name = "EAttack"

    def __init__(self, explainer, run_config, attack_size, attack_inds, targeted, max_rewire, random_rewire,
                 attack_edges, attack_features, edge_mode, features_mode, **kwargs):
        super().__init__(**kwargs)
        self.explainer = explainer
        self.run_config = run_config
        # self.mode = mode
        self.mode = getattr(run_config, CONFIG_OBJ).mode
        self.params = getattr(getattr(run_config, CONFIG_OBJ).kwargs, CONFIG_OBJ).to_dict()
        self.attack_size = attack_size
        self.targeted = targeted
        self.max_rewire = max_rewire
        self.attack_inds = attack_inds
        self.random_rewire = random_rewire
        self.attack_edges = attack_edges
        self.attack_features = attack_features
        self.edge_mode = edge_mode
        self.features_mode = features_mode


    def attack(self, model_manager, gen_dataset, mask_tensor):
        """
        Now edge modification designed in order to get list of attacked nodes that consist of one specific node TODO - refactor
        Feature attack to be developed in order to attack targetted node
        """

        assert self.attack_edges or self.attack_features

        explanations = []

        # get explanations
        for i in tqdm(range(len(self.attack_inds))):
            self.params['element_idx'] = self.attack_inds[i]
            self.explainer.pbar = ProgressBar(None, "er", desc=f'{self.explainer.name} explaining')
            self.explainer.run(self.mode, self.params, finalize=True)
            explanation = copy.deepcopy(self.explainer.explanation.dictionary['data'])
            explanations.append(explanation)

        edge_index = gen_dataset.dataset.data.edge_index.tolist()
        edge_index_set = set([(u, v) for u, v in zip(edge_index[0], edge_index[1])])

        if self.attack_edges:
            cnt = 0

            for i, n in enumerate(self.attack_inds):
                if self.edge_mode == 'remove':
                    for e in explanations[i]['edges'].keys():
                        u, v = map(int, e.split(','))
                        if u != n and v != n: # not remove within 1-hop
                            # TODO check with discard of (v, u) too
                            cnt += 1
                            edge_index_set.discard((u, v))
                            # TEST
                            edge_index_set.discard((v, u))
                elif self.edge_mode == 'add':
                    unimportant_nodes = set()
                    important_nodes = set()
                    for (u, v) in zip(edge_index[0], edge_index[1]):
                        if v == n:
                            if f"{u},{v}" not in explanations[i]['edges'].keys():
                                unimportant_nodes.add(u)
                            else:
                                important_nodes.add(u)
                        elif u == n:
                            if f"{u},{v}" not in explanations[i]['edges'].keys():
                                unimportant_nodes.add(v)
                            else:
                                important_nodes.add(v)
                        # if ((v == n and f"{u},{v}" not in explanations[i]['edges'].keys()) and
                        #         f"{v},{u}" not in explanations[i]['edges'].keys()):
                        #     unimportant_nodes.add(u)
                        # elif v == n:
                        #     important_nodes.add(u)
                    unimportant_nodes = list(unimportant_nodes)
                    # TEST
                    edges = [(u, v) for u, v in zip(edge_index[0], edge_index[1]) if u == n or v == n]
                    #print(len(edges))
                    if len(unimportant_nodes) == 0:
                        continue
                    for e in explanations[i]['edges'].keys():
                        u, v = map(int, e.split(','))
                        if v in important_nodes and u != n:
                            new_node = random.sample(unimportant_nodes, 1)
                            edge_index_set.add((u, new_node[0]))
                            cnt += 1
                elif self.edge_mode == 'rewire':
                    max_rewire = self.max_rewire
                    for (u, v) in zip(edge_index[0], edge_index[1]):
                        if u != n and v != n and f"{u},{v}" in explanations[i]['edges'].keys():
                            edge_index_set.discard((u, v))
                            edge_index_set.discard((v, u))
                            if (u, n) not in edge_index_set:
                                cnt += 1
                                # edge_index_set.add((u, n))
                                # edge_index_set.add((n, u))
                                max_rewire -= 1
                            elif (v, n) not in edge_index_set:
                                cnt += 1
                                # edge_index_set.add((v, n))
                                # edge_index_set.add((n, v))
                                max_rewire -= 1
                        if max_rewire <= 0:
                            break

                # Update dataset edges
                edge_index_new = [[], []]
                for (u, v) in edge_index_set:
                    edge_index_new[0].append(u)
                    edge_index_new[1].append(v)
                edge_index_new = torch.tensor(edge_index_new, dtype=torch.int64)
                gen_dataset.dataset.data.edge_index = edge_index_new

            print(cnt)

        if self.attack_features:
            cnt = 0
            for i, n in enumerate(self.attack_inds):
                if self.features_mode == 'reverse':
                    # get 2-hop
                    hop_1 = set()
                    hop_2 = set()
                    for (u, v) in edge_index_set:
                        if u == n:
                            hop_1.add(v)
                        elif v == n:
                            hop_1.add(u)
                    for (u, v) in edge_index_set:
                        if u in hop_1 and v != n and v not in hop_1:
                            hop_2.add(v)
                        elif v in hop_1 and u != n and u not in hop_1:
                            hop_2.add(u)

                    #get features to be reversed
                    #f_mask = torch.zeros_like(gen_dataset.dataset.data.x.shape[0])
                    f_inds = []
                    for f, v in explanations[i]['nodes'].items():
                        if v:
                            f_inds.append(int(f))
                            cnt += 1
                    f_inds = torch.tensor(f_inds)

                    if not f_inds.numel():
                        continue

                    xor_mask = torch.ones_like(f_inds)

                    for i in hop_2.union(hop_1).union(set([int(n)])):
                        gen_dataset.dataset.data.x[i, f_inds] = torch.logical_xor(gen_dataset.dataset.data.x[i, f_inds], xor_mask).float()
                elif self.features_mode == 'drop':
                    pass
            print(cnt)

        return gen_dataset
