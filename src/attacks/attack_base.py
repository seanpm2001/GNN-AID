from base.datasets_processing import GeneralDataset


class Attacker:
    name = "Attacker"

    def __init__(self, gen_dataset: GeneralDataset, model):
        self.gen_dataset = gen_dataset
        self.model = model

    def attack(self):
        pass

    def save(self, path):
        pass

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        return False
