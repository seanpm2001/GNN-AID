import torch

from defense.defense_base import Defender


class EvasionDefender(Defender):
    def __init__(self, **kwargs):
        super().__init__()

    def pre_batch(self, **kwargs):
        pass

    def post_batch(self, **kwargs):
        pass


class EmptyEvasionDefender(EvasionDefender):
    name = "EmptyEvasionDefender"

    def pre_batch(self, **kwargs):
        pass

    def post_batch(self, **kwargs):
        pass


class GradientRegularizationDefender(EvasionDefender):
    name = "GradientRegularizationDefender"

    def __init__(self, regularization_strength=0.1):
        super().__init__()
        self.regularization_strength = regularization_strength

    def post_batch(self, model_manager, batch, loss, **kwargs):
        batch.x.requires_grad = True
        outputs = model_manager.gnn(batch.x, batch.edge_index)
        loss_loc = model_manager.loss_function(outputs, batch.y)
        gradients = torch.autograd.grad(outputs=loss_loc, inputs=batch.x,
                                        grad_outputs=torch.ones_like(loss_loc),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = torch.sum(gradients ** 2)
        return {"loss": loss + self.regularization_strength * gradient_penalty}


# TODO Kirill, add code in pre_batch
class QuantizationDefender(EvasionDefender):
    name = "QuantizationDefender"

    def __init__(self, qbit=8):
        super().__init__()
        self.regularization_strength = qbit

    def pre_batch(self, **kwargs):
        pass
