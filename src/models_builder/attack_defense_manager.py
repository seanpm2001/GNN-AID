
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

    @staticmethod
    def available_attacker():
        pass

    @staticmethod
    def available_defender():
        pass
