from attacks.evasion_attacks import EvasionAttacker
from aux.configs import CONFIG_OBJ
from explainers.explainer import ProgressBar
from typing import Dict, Optional


class EAttack(EvasionAttacker):
    name = "EAttack"

    def __init__(self, explainer, run_config, attack_budget, **kwargs):
        super().__init__(**kwargs)
        self.explainer = explainer
        self.run_config = run_config
        # self.mode = mode
        self.mode = getattr(run_config, CONFIG_OBJ).mode
        self.params = getattr(getattr(run_config, CONFIG_OBJ).kwargs, CONFIG_OBJ).to_dict()
        self.attack_budget = attack_budget


    def attack(self, model_manager, gen_dataset, mask_tensor):

        # Get explanation
        self.explainer.pbar = ProgressBar(None, "er", desc=f'{self.explainer.name} explaining')
        self.explainer.run(self.mode, self.params, finalize=True)
        explanation = self.explainer.explanation

        # Perturb graph via explanation
        # V 0.1 - Random rewire
        if 'edges' in explanation.dictionary['data'].keys():
            for i in range(self.attack_budget['edges']):
                pass
        if 'nodes' in explanation.dictionary['data'].keys():
            for i in range(self.attack_budget['nodes']):
                break
        if 'features' in explanation.dictionary['data'].keys():
            for i in range(self.attack_budget['features']):
                break
        return gen_dataset