from abc import abstractmethod, ABC


class Defender(ABC):
    name = "Defender"

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
