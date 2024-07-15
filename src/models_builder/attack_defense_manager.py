
for pack in [
]:
    try:
        __import__(pack + '.out')
    except ImportError:
        print(f"Couldn't import Explainer from {pack}")


class AttackAndDefenseManager:
    def __init__(
            self
    ):
        pass

    def conduct_experiment(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def result_path(self):
        pass

    def set_poison_attacker(self):
        pass

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
