from attacks.attack_base import PoisonAttacker, EvasionAttacker, MIAttacker
from aux.configs import ConfigPattern, PoisonAttackConfig, CONFIG_OBJ, EvasionAttackConfig, MIAttackConfig, \
    PoisonDefenseConfig, EvasionDefenseConfig, MIDefenseConfig
from aux.utils import POISON_ATTACK_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH, MI_ATTACK_PARAMETERS_PATH, \
    POISON_DEFENSE_PARAMETERS_PATH, EVASION_DEFENSE_PARAMETERS_PATH, MI_DEFENSE_PARAMETERS_PATH
from defense.defense_base import PoisonDefender, EvasionDefender, MIDefender

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
        self.mi_defender = None
        self.mi_defense_name = None
        self.mi_defense_config = None
        self.evasion_defender = None
        self.evasion_defense_name = None
        self.evasion_defense_config = None
        self.poison_defense_name = None
        self.poison_defense_config = None
        self.poison_defender = None
        self.mi_attack_config = None
        self.mi_attacker = None
        self.mi_attack_name = None
        self.evasion_attack_config = None
        self.evasion_attacker = None
        self.evasion_attack_name = None
        self.poison_attack_name = None
        self.poison_attacker = None
        self.poison_attack_config = None

        self.poison_attack_flag = False
        self.evasion_attack_flag = False
        self.mi_attack_flag = False
        self.poison_defense_flag = False
        self.evasion_defense_flag = False
        self.mi_defense_flag = False

        self.gen_dataset = gen_dataset
        self.gnn = gnn_manager.gnn
        self.model_manager = gnn_manager
        self.gnn_model_path = gnn_manager.model_path_info()

    def conduct_experiment(self):
        if self.poison_attacker is not None and self.poison_attack_flag:
            self.poison_attacker.attack()
        if self.poison_defender is not None and self.poison_defense_flag:
            self.poison_defender.defense()
        if self.mi_defender is not None and self.mi_defense_flag:
            self.mi_defender.defense()
        if self.evasion_defender is not None and self.evasion_defense_flag:
            self.evasion_defender.defense()
        if self.evasion_attacker is not None and self.evasion_attack_flag:
            self.evasion_attacker.attack()
        if self.mi_attacker is not None and self.poison_attack_flag:
            self.mi_attacker.attack()

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
        poison_attack_kwargs = getattr(self.poison_attack_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in PoisonAttacker.__subclasses__()}
        klass = name_klass[self.poison_attack_name]
        self.poison_attacker = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            **poison_attack_kwargs
        )
        self.poison_attack_flag = True

    def set_evasion_attacker(self, evasion_attack_config=None, evasion_attack_name: str = None):
        if evasion_attack_config is None:
            if evasion_attack_name is None:
                raise Exception("if evasion_attack_config is None, evasion_attack_name must be defined")
            evasion_attack_config = ConfigPattern(
                _class_name=evasion_attack_name,
                _import_path=EVASION_ATTACK_PARAMETERS_PATH,
                _config_class="EvasionAttackConfig",
                _config_kwargs={}
            )
        elif isinstance(evasion_attack_config, EvasionAttackConfig):
            if evasion_attack_name is None:
                raise Exception("if evasion_attack_config is None, evasion_attack_name must be defined")
            evasion_attack_config = ConfigPattern(
                _class_name=evasion_attack_name,
                _import_path=EVASION_ATTACK_PARAMETERS_PATH,
                _config_class="EvasionAttackConfig",
                _config_kwargs=evasion_attack_config.to_saveable_dict(),
            )
        self.evasion_attack_config = evasion_attack_config
        if evasion_attack_name is None:
            evasion_attack_name = self.evasion_attack_config._class_name
        elif evasion_attack_name != self.evasion_attack_config._class_name:
            raise Exception(f"evasion_attack_name and self.evasion_attack_config._class_name should be eqequal, "
                            f"but now evasion_attack_name is {evasion_attack_name}, "
                            f"self.evasion_attack_config._class_name is {self.evasion_attack_config._class_name}")
        self.evasion_attack_name = evasion_attack_name
        evasion_attack_kwargs = getattr(self.evasion_attack_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in EvasionAttacker.__subclasses__()}
        klass = name_klass[self.evasion_attack_name]
        self.evasion_attacker = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            **evasion_attack_kwargs
        )
        self.evasion_attack_flag = True

    def set_mi_attacker(self, mi_attack_config=None, mi_attack_name: str = None):
        if mi_attack_config is None:
            if mi_attack_name is None:
                raise Exception("if mi_attack_config is None, mi_attack_name must be defined")
            mi_attack_config = ConfigPattern(
                _class_name=mi_attack_name,
                _import_path=MI_ATTACK_PARAMETERS_PATH,
                _config_class="MIAttackConfig",
                _config_kwargs={}
            )
        elif isinstance(mi_attack_config, MIAttackConfig):
            if mi_attack_name is None:
                raise Exception("if mi_attack_config is None, mi_attack_name must be defined")
            mi_attack_config = ConfigPattern(
                _class_name=mi_attack_name,
                _import_path=MI_ATTACK_PARAMETERS_PATH,
                _config_class="MIAttackConfig",
                _config_kwargs=mi_attack_config.to_saveable_dict(),
            )
        self.mi_attack_config = mi_attack_config
        if mi_attack_name is None:
            mi_attack_name = self.mi_attack_config._class_name
        elif mi_attack_name != self.mi_attack_config._class_name:
            raise Exception(f"mi_attack_name and self.mi_attack_config._class_name should be eqequal, "
                            f"but now mi_attack_name is {mi_attack_name}, "
                            f"self.mi_attack_config._class_name is {self.mi_attack_config._class_name}")
        self.mi_attack_name = mi_attack_name
        mi_attack_kwargs = getattr(self.mi_attack_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in MIAttacker.__subclasses__()}
        klass = name_klass[self.mi_attack_name]
        self.mi_attacker = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            **mi_attack_kwargs
        )
        self.mi_attack_flag = True

    def set_poison_defender(self, poison_defense_config=None, poison_defense_name: str = None):
        if poison_defense_config is None:
            if poison_defense_name is None:
                raise Exception("if poison_defense_config is None, poison_defense_name must be defined")
            poison_defense_config = ConfigPattern(
                _class_name=poison_defense_name,
                _import_path=POISON_DEFENSE_PARAMETERS_PATH,
                _config_class="PoisonDefenseConfig",
                _config_kwargs={}
            )
        elif isinstance(poison_defense_config, PoisonDefenseConfig):
            if poison_defense_name is None:
                raise Exception("if poison_defense_config is None, poison_defense_name must be defined")
            poison_defense_config = ConfigPattern(
                _class_name=poison_defense_name,
                _import_path=POISON_DEFENSE_PARAMETERS_PATH,
                _config_class="PoisonDefenseConfig",
                _config_kwargs=poison_defense_config.to_saveable_dict(),
            )
        self.poison_defense_config = poison_defense_config
        if poison_defense_name is None:
            poison_defense_name = self.poison_defense_config._class_name
        elif poison_defense_name != self.poison_defense_config._class_name:
            raise Exception(f"poison_defense_name and self.poison_defense_config._class_name should be eqequal, "
                            f"but now poison_defense_name is {poison_defense_name}, "
                            f"self.poison_defense_config._class_name is {self.poison_defense_config._class_name}")
        self.poison_defense_name = poison_defense_name
        poison_defense_kwargs = getattr(self.poison_defense_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in PoisonDefender.__subclasses__()}
        klass = name_klass[self.poison_defense_name]
        self.poison_defender = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            **poison_defense_kwargs
        )
        self.poison_defense_flag = True

    def set_evasion_defender(self, evasion_defense_config=None, evasion_defense_name: str = None):
        if evasion_defense_config is None:
            if evasion_defense_name is None:
                raise Exception("if evasion_defense_config is None, evasion_defense_name must be defined")
            evasion_defense_config = ConfigPattern(
                _class_name=evasion_defense_name,
                _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
                _config_class="EvasionDefenseConfig",
                _config_kwargs={}
            )
        elif isinstance(evasion_defense_config, EvasionDefenseConfig):
            if evasion_defense_name is None:
                raise Exception("if evasion_defense_config is None, evasion_defense_name must be defined")
            evasion_defense_config = ConfigPattern(
                _class_name=evasion_defense_name,
                _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
                _config_class="EvasionDefenseConfig",
                _config_kwargs=evasion_defense_config.to_saveable_dict(),
            )
        self.evasion_defense_config = evasion_defense_config
        if evasion_defense_name is None:
            evasion_defense_name = self.evasion_defense_config._class_name
        elif evasion_defense_name != self.evasion_defense_config._class_name:
            raise Exception(f"evasion_defense_name and self.evasion_defense_config._class_name should be eqequal, "
                            f"but now evasion_defense_name is {evasion_defense_name}, "
                            f"self.evasion_defense_config._class_name is {self.evasion_defense_config._class_name}")
        self.evasion_defense_name = evasion_defense_name
        evasion_defense_kwargs = getattr(self.evasion_defense_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in EvasionDefender.__subclasses__()}
        klass = name_klass[self.evasion_defense_name]
        self.evasion_defender = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            **evasion_defense_kwargs
        )
        self.evasion_defense_flag = True

    def set_mi_defender(self, mi_defense_config=None, mi_defense_name: str = None):
        if mi_defense_config is None:
            if mi_defense_name is None:
                raise Exception("if mi_defense_config is None, mi_defense_name must be defined")
            mi_defense_config = ConfigPattern(
                _class_name=mi_defense_name,
                _import_path=MI_DEFENSE_PARAMETERS_PATH,
                _config_class="MIDefenseConfig",
                _config_kwargs={}
            )
        elif isinstance(mi_defense_config, MIDefenseConfig):
            if mi_defense_name is None:
                raise Exception("if mi_defense_config is None, mi_defense_name must be defined")
            mi_defense_config = ConfigPattern(
                _class_name=mi_defense_name,
                _import_path=MI_DEFENSE_PARAMETERS_PATH,
                _config_class="MIDefenseConfig",
                _config_kwargs=mi_defense_config.to_saveable_dict(),
            )
        self.mi_defense_config = mi_defense_config
        if mi_defense_name is None:
            mi_defense_name = self.mi_defense_config._class_name
        elif mi_defense_name != self.mi_defense_config._class_name:
            raise Exception(f"mi_defense_name and self.mi_defense_config._class_name should be eqequal, "
                            f"but now mi_defense_name is {mi_defense_name}, "
                            f"self.mi_defense_config._class_name is {self.mi_defense_config._class_name}")
        self.mi_defense_name = mi_defense_name
        mi_defense_kwargs = getattr(self.mi_defense_config, CONFIG_OBJ).to_dict()

        name_klass = {e.name: e for e in MIDefender.__subclasses__()}
        klass = name_klass[self.mi_defense_name]
        self.mi_defender = klass(
            self.gen_dataset, model=self.gnn,
            # device=self.device,
            # device=device("cpu"),
            **mi_defense_kwargs
        )
        self.mi_defense_flag = True

    def set_all(self):
        pass

    @staticmethod
    def available_attacker():
        pass

    @staticmethod
    def available_defender():
        pass
