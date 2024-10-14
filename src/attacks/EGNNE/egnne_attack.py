from attacks.evasion_attacks import EvasionAttacker
from aux.configs import CONFIG_OBJ
from explainers.explainer import ProgressBar
from typing import Dict, Optional


class EAttack(EvasionAttacker):
    name = "EAttack"

    def __init__(self, explainer, run_config, attack_budget_node, attack_budget_edge, attack_budget_feature, **kwargs):
        super().__init__(**kwargs)
        self.explainer = explainer
        self.run_config = run_config
        # self.mode = mode
        self.mode = getattr(run_config, CONFIG_OBJ).mode
        self.params = getattr(getattr(run_config, CONFIG_OBJ).kwargs, CONFIG_OBJ).to_dict()
        self.attack_budget_node = attack_budget_node
        self.attack_budget_edge = attack_budget_edge
        self.attack_budget_feature = attack_budget_feature


    def attack(self, model_manager, gen_dataset, mask_tensor):

        # Get explanation
        self.explainer.pbar = ProgressBar(None, "er", desc=f'{self.explainer.name} explaining')
        self.explainer.run(self.mode, self.params, finalize=True)
        explanation = self.explainer.explanation

        # Perturb graph via explanation
        # V 0.1 - Random rewire
        if 'edges' in explanation.dictionary['data'].keys():
            for i in range(self.attack_budget_edge):
                pass
        if 'nodes' in explanation.dictionary['data'].keys():
            for i in range(self.attack_budget_node):
                break
        if 'features' in explanation.dictionary['data'].keys():
            for i in range(self.attack_budget_feature):
                break
        return gen_dataset