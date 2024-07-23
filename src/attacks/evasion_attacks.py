from attacks.attack_base import Attacker


class EvasionAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class EmptyEvasionAttacker(EvasionAttacker):
    name = "EmptyEvasionAttacker"

    def attack(self, **kwargs):
        pass
