import unittest
import numpy as np
import torch
import networkx as nx
import os
print(os.getcwd())
import cmkt
# class TestCMKT(unittest.TestCase):
#
#     def test_concept_map_creation(self):
#         # Define a dynamic set of concepts and edges
#         concepts = ["C1", "C2", "C3", "C4", "C5", "C6"]
#         edges = [
#             ("C1", "C3"),
#             ("C2", "C3"),
#             ("C3", "C6"),
#             ("C3", "C4"),
#             ("C4", "C5")
#         ]
#         concept_map = create_concept_map(concepts, edges)
#         self.assertEqual(len(concept_map.nodes), len(concepts))
#         self.assertEqual(len(concept_map.edges), len(edges))
#
#         # Verify the graph structure
#         self.assertTrue(nx.is_directed_acyclic_graph(concept_map))
#         self.assertIn("C1", concept_map.nodes)
#         self.assertIn(("C1", "C3"), concept_map.edges)
#
#     def test_knowledge_state_initialization(self):
#         from cmkt import initialize_knowledge_states
#         num_students = 10
#         num_concepts = 6
#         knowledge_states = initialize_knowledge_states(num_students, num_concepts)
#         self.assertEqual(knowledge_states.shape, (num_students, num_concepts))
#
#     def test_update_knowledge_tracing(self):
#         concepts = ["C1", "C2", "C3", "C4", "C5", "C6"]
#         edges = [
#             ("C1", "C3"),
#             ("C2", "C3"),
#             ("C3", "C6"),
#             ("C3", "C4"),
#             ("C4", "C5")
#         ]
#         concept_map = create_concept_map(concepts, edges)
#         knowledge_states = initialize_knowledge_states(2, len(concepts))
#         responses = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
#         updated_states = update_kt(knowledge_states, responses, concept_map)
#         self.assertEqual(updated_states.shape, (2, len(concepts)))
#
#     def test_dag_gru_forward(self):
#         from cmkt import DAG_GRU
#         model = DAG_GRU(input_size=6, hidden_size=3)
#         x = torch.rand(10, 1, 6)
#         h = torch.zeros(1, 1, 3)
#         output, hidden = model(x, h)
#         self.assertEqual(output.shape, (10, 1, 3))
#         self.assertEqual(hidden.shape, (1, 1, 3))
#
# if __name__ == '__main__':
#     unittest.main()
