from abc import abstractmethod, ABC


class Attacker(ABC):
    name = "Attacker"

    @staticmethod
    def check_availability():
        return False

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def save(self, path):
        pass
