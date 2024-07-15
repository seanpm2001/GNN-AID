from base.datasets_processing import GeneralDataset


class Defender:
    name = "Defender"

    def __init__(self, gen_dataset: GeneralDataset, model):
        self.gen_dataset = gen_dataset
        self.model = model

    def defense(self):
        pass

    def save(self, path):
        pass

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        return False
