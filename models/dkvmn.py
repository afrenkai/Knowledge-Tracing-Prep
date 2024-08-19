
# source: https://arxiv.org/abs/1611.08108

import numpy as np

class DKVMN:
    def __init__(self, mem_size, mem_state_dim, key_mem_state_dim) :
        super(DKVMN, self).__init__()
        self.memory_size = mem_size
        self.memory_state_dimension = mem_state_dim
        self.key_memory_state_dimension = key_mem_state_dim

        ## init ##
        # random array with dim mem_size @ key_memory_state_dim
        self.key_memory = np.random.randn(self.memory_size, self.key_memory_state_dimension)
        self.value_memory = np.random.randn(self.memory_size, self.memory_state_dimension)

        self.input_embedding_matrix = np.random.randn(2 * self.memory_size, self.memory_state_dimension)
        self.question_embedding_matrix = np.random.randn(self.memory_size, self.memory_state_dimension)
        self.linear_weights = np.random.randn(self.memory_state_dimension * 2, 1)

    def embedding_lookup(self, indices, embedding_matrix):
        return embedding_matrix[indices]

    def forward(self, question, answer):
        input_embedding = self.embedding_lookup(question * 2 + answer, self.input_embedding_matrix)
        question_embedding = self.embedding_lookup(question, self.question_embedding_matrix)

        #compute corr weights

        correlation_weight = np.dot(question_embedding, self.key_memory.T)
        #softmax activation func
        correlation_weight = self.softmax(correlation_weight)

        read_content = np.dot(correlation_weight, self.value_memory)

        input_embedding_expanded = np.expand_dims(input_embedding, axis=1)
        correlation_weight_expanded = np.expand_dims(correlation_weight, axis=2)
        self.value_memory += np.sum(correlation_weight_expanded * input_embedding_expanded, axis = 0)

        output = np.concatenate([read_content, input_embedding], axis = 1)
        output = np.dot(output, self.linear_weights)
        # final activation func
        output = self.sigmoid(output)

        return output

    def softmax(self, x):
        e_x = np.exp(x -np.max(x))
        return e_x / e_x.sum(axis = 1, keepdims = True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



# Tests

def test_init():
    mem_size = 50
    mem_state_dim = 100
    key_mem_state_dim = 75

    dkvmn = DKVMN(mem_size, mem_state_dim, key_mem_state_dim)
    assert dkvmn.key_memory.shape == (mem_size, key_mem_state_dim)
    assert dkvmn.value_memory.shape == (mem_size, mem_state_dim)
    assert dkvmn.input_embedding_matrix.shape == (2*mem_size, mem_state_dim)
    assert dkvmn.question_embedding_matrix.shape == (mem_size, mem_state_dim)
    assert dkvmn.linear_weights.shape == (mem_state_dim * 2, 1)

    print('DKVMN is initialized')


test_init()

