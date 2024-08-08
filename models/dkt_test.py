import unittest
import numpy
from dkt import *
class TestDKT(unittest.TestCase):
    def setup(self):
        self.num_skills = 100
        self.hidden_size = 50
        self.sequence_length = 10
        self.batch_size = 5

        self.dkt = DeepKnowledgeTracing(self.num_skills, self.hidden_size)
        self.sequences = [np.random.randint(0,2, (self.sequence_length, self.num_skills * 2)) for _ in range(self.batch_size)]
        self.labels = [np.random.randint(0, 2, (self.sequence_length, self.num_skills)) for _ in range(self.batch_size)]

    def test_fwd_pass(self):
        pred = self.dkt.predict(self.sequences[0])
        self.assertEqual(predicitons.shape, (self.sequence_length, self.hidden_size))
        self.assertTrue(predictions >= 0).all() and (predictions <= 1).all(), "Predictions should be between 0 and 1 (we are in a PDF)"
        

if __name__ == '__main__':
    unittest.main()