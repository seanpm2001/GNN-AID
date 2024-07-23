from defense.defense_base import Defender


class EvasionDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass


class EmptyEvasionDefender(EvasionDefender):
    name = "EmptyEvasionDefender"

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass


class GradientRegularizationDefender(EvasionDefender):
    name = "GradientRegularizationDefender"

    def __init__(self, regularization_strength=0.1):
        super().__init__()
        self.regularization_strength = regularization_strength

    def post_epoch(self, model_manager, data, labels):
        # Implement gradient regularization logic
        for param in model_manager.model.get_parameters():
            if param.grad is not None:
                param.grad += self.regularization_strength * param
