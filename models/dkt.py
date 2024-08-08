import numpy as np

'''NOTE: Using Deep Knowledge Tracing by Piech, et. al. (https://doi.org/10.48550/arXiv.1506.05908) 
as well as np documentation https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html to generate like zero arrays for derivatives.
using https://en.wikipedia.org/wiki/Long_short-term_memory for general theory and applications of LSTM (which DKT obviously dpends on. )
'''


class LSTM():
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size




        ####### Weight Initialization #######

        # Weight for input gate

        self.W_i = np.random.randn(hidden_size, input_size)
        self.U_i = np.random.randn(hidden_size, hidden_size)
        self.b_i = np.zeros((hidden_size, 1))

        
        # forget gate weights 

        self.W_f = np.random.randn(hidden_size, input_size)
        self.U_f = np.random.randn(hidden_size, hidden_size)
        self.b_f = np.zeros((hidden_size, 1))

        # output gate weights

        self.W_o = np.random.randn(hidden_size, input_size)
        self.U_o = np.random.randn(hidden_size, hidden_size)
        self.b_o = np.zeros((hidden_size, 1))

        # cell state weights

        self.W_c = np.random.randn(hidden_size, input_size)
        self.U_c = np.random.randn(hidden_size, hidden_size)
        self.b_c = np.zeros((hidden_size, 1))

        # init hidden state

        self.h = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))

        ####### Activation Functions / Derivatives #######



        # sigmoid activator func, or 1/(1 + e^-x)
        def sigmoid (self, x):
            return 1/ (1 + np.exp(-x))
        # sigoid prime. fun little diversion https://math.stackexchange.com/questions/4620347/why-is-the-derivative-of-the-sigmoid-function-equivalent-to-x-1-x
        def sigmoid_derivative(self, x):
            return x* (1-x)
        #tanh activation function (we do a little cheating (im too lazy to write out the fn))
        def tanh(self, x):
            return np.tanh(x)
        
        # tanh prime https://socratic.org/questions/what-is-the-derivative-of-tanh-x

        def tanh_derivative (self, x):
            return 1- self.tanh(x) ** 2
        
        
        ####### Forward Step #######

        # forward pass
        def forward(self, x):
            outputs = [] 
            self.f, self.i, self.o, self.c_til = [],[],[],[]
            self.h_seq, self.c_seq = [], []

            for t in range (len(x)):
                x_tr = x[t].reshape(-1, 1) # transposition

                # forget gate fwd pass (sigmoid(\cdot W_f (current weight matrix of forget gate), X^T (x transposed)) + \cdot U_f (weight matrix of previous state in forget gate), current hidden state ) + b_f (bias for forget gate)
                f_t = self.sigmoid(np.dot(self.W_f, x_tr) + np.dot(self.U_f, self.h) + self.b_f) 
                self.f.append(f_t)
                #input gate fwd pass
                i_t = self.sigmoid(np.dot(self.W_i, x_tr) + np.dot(self.U_i, self.h) + self.b_i)
                self.i.append(i_t)

                #candidate cell state
                c_til_t = self.tanh(np.dot (self.W_c, x_tr) + np.dot(self.U_c, self.h) + self.b_c)
                self.c_til.append(c_til_t)
                # update cell state
                self.c = f_t * self.c + i_t *c_til_t
                self.c._seq.append(self.c.copy())

                #output gate
                o_t = self.sigmoid(np.dot (self.W_o, x_tr) + np.dot(self.U_o, self.h) + self.b_o)
                self.o.append(o_t)

                # update hidden state

                self.h = o_t * self.tanh(self.c)
                # output at time t (the iterator)
                self.h_seq.append(self.h.copy())

             # returns the array of all outputs over each time series stored in the array, converted to np.array   
            return np.array(self.h_seq)
        ####### Backpropogation #######
        
        def backward(self, output_dim, eta):
            dW_i, dU_i, db_i = np.zeros_like(self.W_i), np.zeros_like(self.U_i), np.zeros_like(self.b_i)
            dW_f, dU_f, db_i = np.zeros_like(self.W_f), np.zeros_like(self.U_f), np.zeros_like(self.b_f)
            dW_o, dU_o, db_o = np.zeros_like(self.W_o), np.zeros_like(self.U_o), np.zeros_like(self.b_o)
            dW_c, dU_c, db_c = np.zeros_like(self.W_c), np.zeros_like(self.U_c), np.zeros_like(self.b_c)
            #d(h+1)
            d_h_next = np.zeros_like(self.h)
            d_C_next = np.zeros_like(self.c)

            for t in reversed(range(len(self.x))):
                #del output gate
                x_tr = x[t].reshape(-1,1)
                do_t = output_dim[t].reshape(-1, 1) * self.tanh(self.seq_c[t])
                do_t_raw = do_t * self.sigmoid_derivative(self.o[t])
                dW_o += np.dot(do_t_raw, xt.T)
                # prior state is t-1
                dU_o += np.dot(do_t_raw, self.h_seq[t-1].T if t>0 else np.zeros_like(self.h).T)
                db_o += do_t_raw

                # del cell state
                d_c = output_dim[t].reshape(1, -1)*self.o[t] * self.tanh_derivative(self.c_seq[t]) + d_C_next
                d_C_next += d_c * self.f[t]

                # del forget gate

                df_t = d_c * self.c_seq[t-1] if t> 0 else np.zeros_like(self.c)
                df_t_raw = df_t * self.sigmoid_derivative(self.f[t])
                dW_f += np.dot(df_t_raw, x_tr.T)
                dU_f += np.dot(df_t_raw, self.h_seq[t-1] if t> 0 else np.zeros_like(self.h).T)
                db_f += df_r_raw

                # del input gate

                di_t = d_c * self.c_til[t]
                di_t_raw = di_t * self.sigmoid_derivative(self.i[t])
                dW_i += np.dot(di_t_raw, x_tr.T)
                dU_i += np.dot(di_t_raw, self.h_seq[t-1] if t > 0 else np.zeros.like(self.h).T)
                db_i += di_t_raw

                # del candidate cell state
                dc_til_t = d_c * self.i[t]
                dc_til_t_raw = dc_til_t * self.tanh_derivative(self.c_til[t])
                dW_c += np.dot(dc_til_t_raw, x_tr.T)
                dU_c += np.dot(dc_til_t_raw, self.h_seq[t -1] if t > 0 else np.zeros_like(self.h).T)
                db_c += dc_til_t_raw

                # update hidden state 
                d_h_next = np.dot(self.U_f.T, df_t_raw)+ np.dot(self.U_i.T, di_t_raw) + np.dot(self.U_o.T, do_t_raw) + np.dot(self.U_c.T, dc_til_t_raw)

                #update weights

                self.W_i -= eta * dW_i
                self.U_i -= eta * dU_i
                self.b_i -= eta * db_i
                self.W_f -= eta * dW_f
                self.U_f -= eta * dU_f
                self.b_f -= eta * db_f
                self.W_o -= eta * dW_o
                self.U_o -= eta * dU_o
                self.b_o -= eta * db_o
                self.W_c -= eta * dW_c
                self.U_c -= eta * dU_c
                self.b_c -= eta * db_c
                


            
class DeepKnowledgeTracing():
    def __init__(self, num_skills, hidden_size):
        self.num_skills = num_skills
        self.hidden_size = hidden_size

        self.lstm = LSTM(num_skills * 2, hidden_size)

        def train(self, sequences, labels, epochs, eta):
            for epoch in range (epochs):
                total_loss = 0
                for seq_label in zip(sequences, labels):
                    # fwd pass
                    predictions = self.lstm.forward(seq)

                    # Binary Cross Entropy Loss Calculation
                    loss = -np.sum(label * np.log(predicitons) + (1-label) * np.log(1-predictions))
                    total_loss += loss
                    output_dim = predicitons - label
                    #BackPropogation Through Time (BPTT)
                    self.lstm.backward(output_dim, eta)
                    print (f'Epoch {epoch + 1}/ {epochs}, Loss: {total_loss}')


        def predict(self, sequence):
            return self.lstm.forward(sequence)
