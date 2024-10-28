import numpy as np
import torch
from torch_geometric.utils import subgraph


class NodesExplainerMetric:
    def __init__(self, model, graph, explainer):
        self.model = model
        self.explainer = explainer
        self.graph = graph
        self.x = self.graph.x
        self.edge_index = self.graph.edge_index
        self.nodes_explanations = {}  # explanations cache. node_ind -> explanation
        self.dictionary = {
        }

    def evaluate(self, target_nodes_indices):
        num_targets = len(target_nodes_indices)
        sparsity = 0
        stability = 0
        consistency = 0
        for node_ind in target_nodes_indices:
            self.get_explanation(node_ind)
            sparsity += self.calculate_sparsity(node_ind)
            stability += self.calculate_stability(node_ind)
            consistency += self.calculate_consistency(node_ind)
        fidelity = self.calculate_fidelity(target_nodes_indices)
        self.dictionary["sparsity"] = sparsity / num_targets
        self.dictionary["stability"] = stability / num_targets
        self.dictionary["consistency"] = consistency / num_targets
        self.dictionary["fidelity"] = fidelity
        return self.dictionary

    def calculate_fidelity(self, target_nodes_indices):
        original_answer = self.model.get_answer(self.x, self.edge_index)
        same_answers_count = 0
        for node_ind in target_nodes_indices:
            node_explanation = self.get_explanation(node_ind)
            new_x, new_edge_index, new_target_node = self.filter_graph_by_explanation(
                self.x, self.edge_index, node_explanation, node_ind
            )
            filtered_answer = self.model.get_answer(new_x, new_edge_index)
            matched = filtered_answer[new_target_node] == original_answer[node_ind]
            print(f"Processed fidelity calculation for node id {node_ind}. Matched: {matched}")
            if matched:
                same_answers_count += 1
        fidelity = same_answers_count / len(target_nodes_indices)
        return fidelity

    def calculate_sparsity(self, node_ind):
        explanation = self.get_explanation(node_ind)
        sparsity = 1 - (len(explanation["data"]["nodes"]) + len(explanation["data"]["edges"])) / (
                len(self.x) + len(self.edge_index))
        return sparsity

    def calculate_stability(self, node_ind, feature_change_percent=0.05, node_removal_percent=0.05):
        base_explanation = self.get_explanation(node_ind)
        new_x, new_edge_index = self.perturb_graph(
            self.x, self.edge_index, node_ind, feature_change_percent, node_removal_percent
        )
        perturbed_explanation = self.calculate_explanation(new_x, new_edge_index, node_ind)

        base_explanation_vector, perturbed_explanation_vector = \
            NodesExplainerMetric.calculate_explanation_vectors(base_explanation, perturbed_explanation)

        return euclidean_distance(base_explanation_vector, perturbed_explanation_vector)

    def calculate_consistency(self, node_ind):
        base_explanation = self.get_explanation(node_ind)
        perturbed_explanation = self.calculate_explanation(self.x, self.edge_index, node_ind)
        base_explanation_vector, perturbed_explanation_vector = \
            NodesExplainerMetric.calculate_explanation_vectors(base_explanation, perturbed_explanation)
        return cosine_similarity(base_explanation_vector, perturbed_explanation_vector)

    def calculate_explanation(self, x, edge_index, node_idx, **kwargs):
        self.explainer.evaluate_tensor_graph(x, edge_index, node_idx, **kwargs)
        return self.explainer.explanation.dictionary

    def get_explanation(self, node_ind):
        if node_ind in self.nodes_explanations:
            node_explanation = self.nodes_explanations[node_ind]
        else:
            node_explanation = self.calculate_explanation(self.x, self.edge_index, node_ind)
            self.nodes_explanations[node_ind] = node_explanation
            print(f"Processed explanation calculation for node id {node_ind}.")
        return node_explanation

    @staticmethod
    def parse_explanation(explanation):
        important_nodes = {
            int(node): float(weight) for node, weight in explanation["data"]["nodes"].items()
        }
        important_edges = {
            tuple(map(int, edge.split(','))): float(weight)
            for edge, weight in explanation["data"]["edges"].items()
        }
        return important_nodes, important_edges

    @staticmethod
    def filter_graph_by_explanation(x, edge_index, explanation, target_node):
        important_nodes, important_edges = NodesExplainerMetric.parse_explanation(explanation)
        all_important_nodes = set(important_nodes.keys())
        all_important_nodes.add(target_node)
        for u, v in important_edges.keys():
            all_important_nodes.add(u)
            all_important_nodes.add(v)

        important_node_indices = list(all_important_nodes)
        node_mask = torch.zeros(x.size(0), dtype=torch.bool)
        node_mask[important_node_indices] = True

        new_edge_index, new_edge_weight = subgraph(node_mask, edge_index, relabel_nodes=True)
        new_x = x[node_mask]
        new_target_node = important_node_indices.index(target_node)
        return new_x, new_edge_index, new_target_node

    @staticmethod
    def calculate_explanation_vectors(base_explanation, perturbed_explanation):
        base_important_nodes, base_important_edges = NodesExplainerMetric.parse_explanation(
            base_explanation
        )
        perturbed_important_nodes, perturbed_important_edges = NodesExplainerMetric.parse_explanation(
            perturbed_explanation
        )
        union_nodes = set(base_important_nodes.keys()) | set(perturbed_important_nodes.keys())
        union_edges = set(base_important_edges.keys()) | set(perturbed_important_edges.keys())
        explain_vector_len = len(union_nodes) + len(union_edges)
        base_explanation_vector = np.zeros(explain_vector_len)
        perturbed_explanation_vector = np.zeros(explain_vector_len)
        i = 0
        for expl_node_ind in union_nodes:
            base_explanation_vector[i] = base_important_nodes.get(expl_node_ind, 0)
            perturbed_explanation_vector[i] = perturbed_important_nodes.get(expl_node_ind, 0)
            i += 1
        for expl_edge in union_edges:
            base_explanation_vector[i] = base_important_edges.get(expl_edge, 0)
            perturbed_explanation_vector[i] = perturbed_important_edges.get(expl_edge, 0)
            i += 1
        return base_explanation_vector, perturbed_explanation_vector

    @staticmethod
    def perturb_graph(x, edge_index, node_ind, feature_change_percent, node_removal_percent):
        new_x = x.clone()
        num_nodes = x.shape[0]
        num_features = x.shape[1]
        num_features_to_change = int(feature_change_percent * num_nodes * num_features)
        indices = torch.randint(0, num_nodes * num_features, (num_features_to_change,), device=x.device)
        new_x.view(-1)[indices] = 1.0 - new_x.view(-1)[indices]

        neighbors = edge_index[1][edge_index[0] == node_ind].unique()
        num_nodes_to_remove = int(node_removal_percent * neighbors.shape[0])

        if num_nodes_to_remove > 0:
            nodes_to_remove = neighbors[
                torch.randperm(neighbors.size(0), device=edge_index.device)[:num_nodes_to_remove]
            ]
            mask = ~((edge_index[0] == node_ind).unsqueeze(1) & (edge_index[1].unsqueeze(0) == nodes_to_remove).any(
                dim=0))
            new_edge_index = edge_index[:, mask]
        else:
            new_edge_index = edge_index

        return new_x, new_edge_index


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
