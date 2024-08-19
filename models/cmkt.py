# creation of sample concept map as per sec 4 p b

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List, Tuple, Dict
import unittest

def create_concept_map(concepts: List[str], edges: List[Tuple[str, str]]) -> nx.DiGraph:
    """
    Creates a directed acyclic graph (DAG) representing the concept map.

    Args:
    - concepts: A list of concepts (nodes) in the concept map.
    - edges: A list of tuples representing directed edges between concepts (e.g., ('C1', 'C3')).

    Returns:
    - A directed acyclic graph (DAG) representing the concept map.
    """
    G = nx.DiGraph()
    G.add_nodes_from(concepts)
    G.add_edges_from(edges)
    return G


def initialize_knowledge_states(n_students: int, n_concepts: int) -> np.ndarray:
    return np.random.rand(n_students, n_concepts)


def update_kt(knowledge_states: np.ndarray, res: np.ndarray, concept_map: nx.DiGraph) -> np.ndarray:
    concept_to_index = {concept: idx for idx, concept in enumerate(concept_map.nodes)}

    for sid, resp in enumerate(res):
        for concept in concept_map.nodes:
            concept_idx = concept_to_index[concept]
            for prereq in concept_map.predecessors(concept):
                prereq_idx = concept_to_index[prereq]
                if resp[prereq_idx] > knowledge_states[sid][prereq_idx]:
                    knowledge_states[sid][concept_idx] += 0.1 * resp[prereq_idx]
    return knowledge_states


class DAG_GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DAG_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.gru(x, h)
        return output, hidden


def predict_performance(knowledge_states: np.ndarray, concept_map: nx.DiGraph) -> List[float]:
    predictions = []
    for s_know in knowledge_states:
        prediction = np.dot(s_know, np.ones(len(concept_map.nodes)))
        predictions.append(prediction)
    return predictions


def train_model(model: nn.Module, data: List[Tuple[torch.Tensor, torch.Tensor]], eta: float  = 0.01, epochs: int = 100) -> None:
    optimizer = Adam(model.parameters(), lr = eta)
    for epoch in range(epochs):
        for x, y in data:
            optimizer.zero_grad()
            h = torch.zeros(1, x.size(0), model.gru.hidden_size)
            output, hidden = model(x, h)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()






import unittest
class TestCMKT(unittest.TestCase):

    def test_concept_map_creation(self):
        concepts = ["C1", "C2", "C3", "C4", "C5", "C6"]
        edges = [
            ("C1", "C3"),
            ("C2", "C3"),
            ("C3", "C6"),
            ("C3", "C4"),
            ("C4", "C5")
        ]
        concept_map = create_concept_map(concepts, edges)
        self.assertEqual(len(concept_map.nodes), len(concepts))
        self.assertEqual(len(concept_map.edges), len(edges))
        self.assertTrue(nx.is_directed_acyclic_graph(concept_map))
        self.assertIn("C1", concept_map.nodes)
        self.assertIn(("C1", "C3"), concept_map.edges)

    def test_knowledge_state_initialization(self):
        num_students = 10
        num_concepts = 6
        knowledge_states = initialize_knowledge_states(num_students, num_concepts)
        self.assertEqual(knowledge_states.shape, (num_students, num_concepts))

    def test_update_knowledge_tracing(self):
        concepts = ["C1", "C2", "C3", "C4", "C5", "C6"]
        edges = [
            ("C1", "C3"),
            ("C2", "C3"),
            ("C3", "C6"),
            ("C3", "C4"),
            ("C4", "C5")
        ]
        concept_map = create_concept_map(concepts, edges)
        knowledge_states = initialize_knowledge_states(2, len(concepts))
        responses = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        updated_states = update_kt(knowledge_states, responses, concept_map)
        self.assertEqual(updated_states.shape, (2, len(concepts)))

    def test_dag_gru_forward(self):
        model = DAG_GRU(input_size=6, hidden_size=3)
        x = torch.rand(10, 1, 6)
        h = torch.zeros(1, 1, 3)
        output, hidden = model(x, h)
        self.assertEqual(output.shape, (10, 1, 3))
        self.assertEqual(hidden.shape, (1, 1, 3))

if __name__ == '__main__':
    unittest.main()

