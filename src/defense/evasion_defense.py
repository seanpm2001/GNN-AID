import torch

from defense.defense_base import Defender
from src.aux.utils import import_by_name
from attacks.evasion_attacks import FGSMAttacker
from torch_geometric import data

import copy

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


class AdvTraining(EvasionDefender):
    name = "AdvTraining"

    def __init__(self, epsilon=0.1, attack_type="FGSM", device='cpu'):
        super().__init__()
        assert device is not None, "Please specify 'device'!"
        # if attack_type=="NettackEvasion":
        #     self.attacker = evasion_attacks.NettackEvasionAttacker()
        # elif attack_type=="FGSM":
        #     self.attacker = evasion_attacks.FGSMAttacker(epsilon=epsilon)
        # self.attacker = import_by_name(attack_type, ['attacks.evasion_attacks'])()
        self.attacker = FGSMAttacker(epsilon=epsilon)

    def pre_batch(self, model_manager, batch):
        super().pre_batch(model_manager=model_manager, batch=batch)
        # print(batch)
        gen_data = data.Data()
        gen_data.data = copy.deepcopy(batch)
        # print(gen_data)
        # print(batch.batch)
        attacked_batch = self.attacker.attack(model_manager, gen_data, batch.train_mask).data
        new_batch = self.merge_batches(batch, attacked_batch)
        # print(attacked_batch.x.mean() - batch.x.mean())

        # print(batch)
        batch = new_batch
        # print(batch)
    
    def post_batch(self, model_manager, batch, loss) -> dict:
        super().post_batch(model_manager=model_manager, batch=batch, loss=loss)

    @staticmethod
    def merge_batches(batch1, batch2):
        merged_x = torch.cat([batch1.x, batch2.x], dim=0)

        adj_edge_index = batch2.edge_index + batch1.x.size(0)
        merged_edge_index = torch.cat([batch1.edge_index, adj_edge_index], dim=1)
        
        merged_y = torch.cat([batch1.y, batch2.y], dim=0)

        merged_train_mask = torch.cat([batch1.train_mask, batch2.train_mask], dim=0)
        merged_val_mask = torch.cat([batch1.val_mask, batch2.val_mask], dim=0)
        merged_test_mask = torch.cat([batch1.test_mask, batch2.test_mask], dim=0)

        merged_n_id = torch.cat([batch1.n_id, batch2.n_id], dim=0)
        merged_e_id = torch.cat([batch1.e_id, batch2.e_id], dim=0)

        merged_num_sampled_nodes = batch1.num_sampled_nodes + batch2.num_sampled_nodes
        merged_num_sampled_edges = batch1.num_sampled_edges + batch2.num_sampled_edges
        merged_input_id = torch.cat([batch1.input_id, batch2.input_id], dim=0)
        merged_batch_size = batch1.batch_size + batch2.batch_size
        
        merged_batch = None
        if batch1.batch and batch2.batch:
            merged_batch = torch.cat([batch1.batch, batch2.batch + batch1.batch.max() + 1], dim=0)

        return data.Data(x=merged_x,
                         edge_index=merged_edge_index,
                         y=merged_y,
                         train_mask=merged_train_mask,
                         val_mask=merged_val_mask,
                         test_mask=merged_test_mask,
                         n_id=merged_n_id,
                         e_id=merged_e_id,
                         num_sampled_nodes=merged_num_sampled_nodes,
                         num_sampled_edges=merged_num_sampled_edges,
                         input_id=merged_input_id,
                         batch_size=merged_batch_size,
                         batch=merged_batch)