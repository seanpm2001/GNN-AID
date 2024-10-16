import numpy as np
import random
import torch
import copy

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

    def __init__(self, explainer, run_config, attack_size, attack_inds, targeted, max_rewire, **kwargs):
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


    def attack(self, model_manager, gen_dataset, mask_tensor):

        explanations = []
        if not self.targeted:
            # make sample
            node_inds = [i for i, x in enumerate(mask_tensor) if x]
            # dataset = gen_dataset.dataset.data[mask_tensor]
            num_nodes = len(node_inds)
            self.attacked_node_size = int(num_nodes * self.attack_size)
            self.attack_inds = np.random.choice(node_inds, self.attacked_node_size)

        # get explanations
        for i in tqdm(range(len(self.attack_inds))):
            self.params['element_idx'] = self.attack_inds[i]
            self.explainer.pbar = ProgressBar(None, "er", desc=f'{self.explainer.name} explaining')
            self.explainer.run(self.mode, self.params, finalize=True)
            explanations.append(copy.deepcopy(self.explainer.explanation))

        edge_index = gen_dataset.dataset.data.edge_index.tolist()
        edge_index_set = set([(u, v) for u, v in zip(edge_index[0], edge_index[1])])
        neighbours = {n: set() for n in self.attack_inds}
        neighbours_list = list(neighbours)
        for u, v in  zip(edge_index[0], edge_index[1]):
            if u in neighbours.keys():
                neighbours[u].add(v)
            elif v in neighbours.keys():
                neighbours[v].add(u)
        for i, n in enumerate(self.attack_inds):
            max_rewire = self.max_rewire
            important_edges = sorted(list(explanations[i].dictionary['data']['edges'].items()), key=lambda x: x[1], reverse=True)
            important_edges = [list(map(int, y)) for y in tuple(x[0].split(',') for x in important_edges)]
            #edge_explanation = set([(u,v) for u, v in map(int, explanations[i].dictionary['data']['edges'].split(','))])
            for u, v in important_edges:
                if (u, v) not in edge_index_set:
                    continue
                # if second hop neighbour
                if u in neighbours[n] and v != n:
                    rewire_node = v
                    neigh_node = u
                elif v in neighbours[n] and u != n:
                    neigh_node = v
                    rewire_node = u
                else:
                    continue
                if max_rewire:
                    sample = random.sample(neighbours_list, 2)
                    new_neigh = sample[0] if sample[0] != neigh_node else sample[1]
                    edge_index_set.remove((u, v))
                    edge_index_set.add((rewire_node, new_neigh))
                    max_rewire -= 1
        edge_index_new = [[],[]]
        for (u, v) in edge_index_set:
            edge_index_new[0].append(u)
            edge_index_new[1].append(v)
        edge_index_new = torch.tensor(edge_index_new, dtype=torch.int64)
        gen_dataset.dataset.data.edge_index = edge_index_new
        return gen_dataset

        # # Get explanation
        # self.explainer.pbar = ProgressBar(None, "er", desc=f'{self.explainer.name} explaining')
        # self.explainer.run(self.mode, self.params, finalize=True)
        # explanation = self.explainer.explanation
        #
        # # Perturb graph via explanation
        # # V 0.1 - Random rewire
        # if 'edges' in explanation.dictionary['data'].keys():
        #     for i in range(self.attack_budget_edge):
        #         pass
        # if 'nodes' in explanation.dictionary['data'].keys(): # Not implemented yet
        #     for i in range(self.attack_budget_node):
        #         break
        # if 'features' in explanation.dictionary['data'].keys(): # Not implemented yet
        #     for i in range(self.attack_budget_feature):
        #         break
        # return gen_dataset