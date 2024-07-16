from attacks.attack_base import PoisonAttacker
from aux.configs import ConfigPattern, PoisonAttackConfig, CONFIG_OBJ
from aux.utils import POISON_ATTACK_PARAMETERS_PATH

for pack in [
]:
    try:
        __import__(pack + '.out')
    except ImportError:
        print(f"Couldn't import Explainer from {pack}")


class AttackAndDefenseManager:
    def __init__(
            self,
            gen_dataset,
            gnn_manager
    ):
        self.poison_attack_name = None
        self.poison_attacker = None
        self.poison_attack_config = None

        self.gen_dataset = gen_dataset
        self.gnn = gnn_manager.gnn
        self.model_manager = gnn_manager
        self.gnn_model_path = gnn_manager.model_path_info()

    def conduct_experiment(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def result_path(self):
        pass

    def set_poison_attacker(self, poison_attack_config=None, poison_attack_name: str = None):
        if poison_attack_config is None:
            if poison_attack_name is None:
                raise Exception("if poison_attack_config is None, poison_attack_name must be defined")
            poison_attack_config = ConfigPattern(
                _class_name=poison_attack_name,
                _import_path=POISON_ATTACK_PARAMETERS_PATH,
                _config_class="PoisonAttackConfig",
                _config_kwargs={}
            )
        elif isinstance(poison_attack_config, PoisonAttackConfig):
            if poison_attack_name is None:
                raise Exception("if poison_attack_config is None, poison_attack_name must be defined")
            poison_attack_config = ConfigPattern(
                _class_name=poison_attack_name,
                _import_path=POISON_ATTACK_PARAMETERS_PATH,
                _config_class="PoisonAttackConfig",
                _config_kwargs=poison_attack_config.to_saveable_dict(),
            )
        self.poison_attack_config = poison_attack_config
        if poison_attack_name is None:
            poison_attack_name = self.poison_attack_config._class_name
        elif poison_attack_name != self.poison_attack_config._class_name:
            raise Exception(f"poison_attack_name and self.poison_attack_config._class_name should be eqequal, "
                            f"but now poison_attack_name is {poison_attack_name}, "
                            f"self.poison_attack_config._class_name is {self.poison_attack_config._class_name}")
        self.poison_attack_name = poison_attack_name
        # poison_attack_kwargs = getattr(self.poison_attack_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in PoisonAttacker.__subclasses__()}
        klass = name_klass[self.poison_attack_name]
        self.poison_attacker = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            poison_attack_config=self.poison_attack_config)

    def set_evasion_attacker(self):
        pass

    def set_mi_attacker(self):
        pass

    def set_poison_defender(self):
        pass

    def set_evasion_defender(self):
        pass

    def set_mi_defender(self):
        pass

    def set_all(self):
        pass

    @staticmethod
    def available_attacker():
        pass

    @staticmethod
    def available_defender():
        pass
