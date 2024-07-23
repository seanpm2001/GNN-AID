import torch
import torch.nn.functional as F

from attacks.attack_base import Attacker


class EvasionAttacker(Attacker):
    def __init__(self, **kwargs):
        super().__init__()


class EmptyEvasionAttacker(EvasionAttacker):
    name = "EmptyEvasionAttacker"

    def attack(self, **kwargs):
        pass


class FGSMAttacker(EvasionAttacker):
    name = "FGSM"

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def attack(self, model_manager, gen_dataset, mask_tensor):
        gen_dataset.data.x.requires_grad = True
        output = model_manager.gnn(gen_dataset.data.x, gen_dataset.data.edge_index, gen_dataset.data.batch)
        loss = model_manager.loss_function(output[mask_tensor],
                                           gen_dataset.data.y[mask_tensor])
        model_manager.gnn.zero_grad()
        loss.backward()
        sign_data_grad = gen_dataset.data.x.grad.sign()
        perturbed_data_x = gen_dataset.data.x + self.epsilon * sign_data_grad
        perturbed_data_x = torch.clamp(perturbed_data_x, 0, 1)
        gen_dataset.data.x = perturbed_data_x.detach()
        return gen_dataset
