import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
DANTE_PATH = 'dante.txt'
FILEPATH_1 = 'deamicis.txt'  # should be simple plain text file
FILEPATH_2 = 'deamicis2.txt'

HIDDEN_SIZE = 100  # size of hidden layer of neurons
SEQ_LENGHT = 25  # number of steps to unroll the RNN for. Equivalent concept of mini batch.
EPOCHS = 10
LEARNING_RATE = 1e-1


class LSTM(object):

    def __init__(self, input_size, hidden_size, ouput_size, seq_length=SEQ_LENGHT, num_epochs=EPOCHS,
                 activation_fn=np.tanh,
                 learning_rate=LEARNING_RATE):
        self.input_size = input_size  # vocabulary size
        self.ouput_size = ouput_size  # vocabulary size
        self.hidden_size = hidden_size  # hidden layer size
        self.seq_length = seq_length  # number of time steps, aka batch size
        self.epochs = num_epochs
        self.learning_rate = learning_rate

        # self.set_internal_status()
        self.set_model_parameters()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.tanh_activ = activation_fn

        self.history = [-np.log(1.0 / self.input_size) * self.seq_length]
        self.texts = []

    def reset_hidden_cell_status(self):
        self._hidden_state = np.zeros((self.hidden_size, 1))
        self._cell_state = np.zeros((self.hidden_size, 1))

    def set_model_parameters(self):
        std = (1.0 / np.sqrt(self.input_size + self.hidden_size))  # Xavier initialisation

        # Forget gate
        self.Wf = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * std
        self.bf = np.ones((self.hidden_size, 1))
        # Input gate
        self.Wi = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * std
        self.bi = np.zeros((self.hidden_size, 1))
        # G-gate
        self.Wg = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * std
        self.bg = np.zeros((self.hidden_size, 1))
        # Output gate
        self.Wo = np.random.randn(self.hidden_size, self.hidden_size + self.input_size) * std
        self.bo = np.zeros((self.hidden_size, 1))
        # output
        self.Wy = np.random.randn(self.input_size, self.hidden_size) * \
                  (1.0 / np.sqrt(self.input_size))
        self.by = np.zeros((self.input_size, 1))

    def compile(self, loss, optimiser, char_to_indx, indx_to_char):
        self.loss = loss
        self.optimiser = optimiser
        self.encoding = char_to_indx
        self.decoding = indx_to_char

    def forward(self, input, hidden_previous, cell_previous):
        """
        inputs is list of integers.
        hidden_previous is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        # Empty dictionaries of the type dict[time_step] = value_at_time_t
        x, h, z, c = {}, {}, {}, {}
        f, i, g, o = {}, {}, {}, {}
        y, p = {}, {}

        h[-1] = np.copy(hidden_previous)  # computation of hidden state at time t=0 requires previous value (t=-1)
        c[-1] = np.copy(cell_previous)

        for t in range(len(input)):
            x[t] = one_hot_encoding(self.input_size, input[t])
            z[t] = np.row_stack((h[t - 1], x[t]))

            f[t] = self.sigmoid(np.dot(self.Wf, z[t]) + self.bf)
            i[t] = self.sigmoid(np.dot(self.Wi, z[t]) + self.bi)
            g[t] = self.tanh_activ(np.dot(self.Wg, z[t]) + self.bg)
            o[t] = self.sigmoid(np.dot(self.Wo, z[t]) + self.bo)

            c[t] = f[t] * c[t - 1] + i[t] * g[t]
            h[t] = o[t] * np.tanh(c[t])

            y[t] = np.dot(self.Wy, h[t]) + self.by
            p[t] = self.softmax(y[t])

        cell_dict = {"x": x, "h": h, "z": z, "c": c, "f": f, "i": i, "g": g, "o": o, "y": y, "p": p}

        return cell_dict

    def store_last_internal_cell_status(self, h, c):
        self._hidden_state = h[len(h) - 2]
        self._cell_state = c[len(h) - 2]

    # need to work on backwards

    def backward(self, cell_dict, targets):
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWg = np.zeros_like(self.Wg)
        dWo = np.zeros_like(self.Wo)
        dWy = np.zeros_like(self.Wy)

        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbg = np.zeros_like(self.bg)
        dbo = np.zeros_like(self.bo)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(cell_dict["h"][0])
        dc_next = np.zeros_like(cell_dict["c"][0])
        for t in reversed(range(len(cell_dict["p"]))):
            dy = np.copy(cell_dict["p"][t])
            dy[targets[t]] -= 1

            dWy += np.dot(dy, cell_dict["h"][t].T)
            dby += dy

            dh = np.dot(self.Wy.T, dy)
            dh += dh_next

            do = dh * self.tanh_activ(cell_dict["c"][t])
            da_o = do * cell_dict["o"][t] * (1 - cell_dict["o"][t])
            dWo += np.dot(da_o, cell_dict["z"][t].T)
            dbo += da_o

            dc = dh * cell_dict["o"][t] * (1 - self.tanh_activ(cell_dict["c"][t]) ** 2)
            dc += dc_next

            dg = dc * cell_dict["i"][t]
            da_g = dg * (1 - cell_dict["g"][t] ** 2)
            dWg += np.dot(da_g, cell_dict["z"][t].T)
            dbg += da_g

            di = dc * cell_dict["g"][t]
            da_i = di * cell_dict["i"][t] * (1 - cell_dict["i"][t])
            dWi += np.dot(da_i, cell_dict["z"][t].T)
            dbi += da_i

            df = dc * cell_dict["c"][t - 1]
            da_f = df * cell_dict["f"][t] * (1 - cell_dict["f"][t])
            dWf += np.dot(da_f, cell_dict["z"][t].T)
            dbf += da_f

            dz = (np.dot(self.Wf.T, da_f)
                  + np.dot(self.Wi.T, da_i)
                  + np.dot(self.Wg.T, da_g)
                  + np.dot(self.Wo.T, da_o))

            dh_next = dz[:self.hidden_size, :]
            dc_next = cell_dict["f"][t] * dc

        # for dparam in [dWf, dWi, dWg, dWo, dWy, dbf, dbi, dbg, dbo, dby]:
        #     np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        grads = {"dWf": dWf, "dWi": dWi, "dWg": dWg, "dWo": dWo, "dWy": dWy, "dbf": dbf, "dbi": dbi, "dbg": dbg,
                 "dbo": dbo, "dby": dby}
        return grads

    def update_weights(self, grads, iteration):
        self.optimiser.update_adam_weigths(grads, iteration)

        self.Wf += -self.learning_rate * self.optimiser._M_Wf / (np.sqrt(self.optimiser._S_Wf) + 1e-8)
        self.Wi += -self.learning_rate * self.optimiser._M_Wi / (np.sqrt(self.optimiser._S_Wi) + 1e-8)
        self.Wg += -self.learning_rate * self.optimiser._M_Wg / (np.sqrt(self.optimiser._S_Wg) + 1e-8)
        self.Wo += -self.learning_rate * self.optimiser._M_Wo / (np.sqrt(self.optimiser._S_Wo) + 1e-8)
        self.Wy += -self.learning_rate * self.optimiser._M_Wy / (np.sqrt(self.optimiser._S_Wy) + 1e-8)

        self.bf += -self.learning_rate * self.optimiser._M_bf / (np.sqrt(self.optimiser._S_bf) + 1e-8)
        self.bi += -self.learning_rate * self.optimiser._M_bi / (np.sqrt(self.optimiser._S_bi) + 1e-8)
        self.bg += -self.learning_rate * self.optimiser._M_bg / (np.sqrt(self.optimiser._S_bg) + 1e-8)
        self.bo += -self.learning_rate * self.optimiser._M_bo / (np.sqrt(self.optimiser._S_bo) + 1e-8)
        self.by += -self.learning_rate * self.optimiser._M_by / (np.sqrt(self.optimiser._S_by) + 1e-8)

    def train(self, data):
        # loss at iteration 0: the probability of having n=seq_length randomly chosen character from a dictionary
        # containing i=input_size number of different characters.
        self.reset_hidden_cell_status()
        inputs, targets = chunks_input_target(data, self.seq_length, self.encoding)
        for epoch in range(self.epochs):
            for iteration, (input, target) in enumerate(zip(inputs, targets)):

                # forward
                forward_dict = self.forward(input, self._hidden_state, self._cell_state)
                self.store_last_internal_cell_status(forward_dict['h'], forward_dict['c'])

                # compute time-batch-loss
                iteration_loss = self.loss(forward_dict['p'], target)

                # backward
                grads = self.backward(forward_dict, target)

                # update model
                batch_num = epoch * self.epochs + iteration / self.seq_length + 1
                self.update_weights(grads, batch_num)  # <--- iteration must take into account the epoch number. CHANGE

                # compute total loss
                smooth_loss = self.history[-1] * 0.999 + iteration_loss * 0.001
                if iteration % 100 == 0:
                    print('Epoch: {}, iter {}, loss: {}'.format(epoch, iteration, smooth_loss))
                #     self.texts.append(self.sample(inputs[0], 8, self.decoding))

                self.history.append(smooth_loss)

    def sample(self, seed_ix, n, indx_to_char):

        h = self._hidden_state
        c = self._cell_state
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        txt = str()

        for t in range(n):
            z = np.row_stack((h, x))

            f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
            g = self.tanh_activ(np.dot(self.Wg, z) + self.bg)

            c = f * c + i * g
            o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
            h = o * np.tanh(c)

            y = np.dot(self.Wy, h) + self.by
            p = self.softmax(y)
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            txt += indx_to_char[ix]
            x = p
        return txt


class Adam_opt(object):

    def __init__(self, input_size, hidden_size, ouput_size):
        self.beta1 = 0.9
        self.beta2 = 0.999

        # Momentum weights
        self._M_Wf = np.zeros((hidden_size, hidden_size + input_size))
        self._M_Wi = np.zeros((hidden_size, hidden_size + input_size))
        self._M_Wg = np.zeros((hidden_size, hidden_size + input_size))
        self._M_Wo = np.zeros((hidden_size, hidden_size + input_size))
        self._M_Wy = np.zeros((ouput_size, hidden_size))

        self._M_bf = np.zeros((hidden_size, 1))
        self._M_bi = np.zeros((hidden_size, 1))
        self._M_bg = np.zeros((hidden_size, 1))
        self._M_bo = np.zeros((hidden_size, 1))
        self._M_by = np.zeros((ouput_size, 1))

        # RMS weights
        self._S_Wf = np.zeros((hidden_size, hidden_size + input_size))
        self._S_Wi = np.zeros((hidden_size, hidden_size + input_size))
        self._S_Wg = np.zeros((hidden_size, hidden_size + input_size))
        self._S_Wo = np.zeros((hidden_size, hidden_size + input_size))
        self._S_Wy = np.zeros((ouput_size, hidden_size))

        self._S_bf = np.zeros((hidden_size, 1))
        self._S_bi = np.zeros((hidden_size, 1))
        self._S_bg = np.zeros((hidden_size, 1))
        self._S_bo = np.zeros((hidden_size, 1))
        self._S_by = np.zeros((ouput_size, 1))

    def update_adam_weigths(self, grads, t):
        scal1 = 1 - self.beta1 ** t
        scal2 = 1 - self.beta2 ** t
        # momentum update
        self._M_Wf = (self.beta1 * self._M_Wf + (1 - self.beta1) * grads["dWf"]) / scal1
        self._M_Wi = (self.beta1 * self._M_Wi + (1 - self.beta1) * grads["dWi"]) / scal1
        self._M_Wg = (self.beta1 * self._M_Wg + (1 - self.beta1) * grads["dWg"]) / scal1
        self._M_Wo = (self.beta1 * self._M_Wo + (1 - self.beta1) * grads["dWo"]) / scal1
        self._M_Wy = (self.beta1 * self._M_Wy + (1 - self.beta1) * grads["dWy"]) / scal1

        self._M_bf = (self.beta1 * self._M_bf + (1 - self.beta1) * grads["dbf"]) / scal1
        self._M_bi = (self.beta1 * self._M_bi + (1 - self.beta1) * grads["dbi"]) / scal1
        self._M_bg = (self.beta1 * self._M_bg + (1 - self.beta1) * grads["dbg"]) / scal1
        self._M_bo = (self.beta1 * self._M_bo + (1 - self.beta1) * grads["dbo"]) / scal1
        self._M_by = (self.beta1 * self._M_by + (1 - self.beta1) * grads["dby"]) / scal1

        # rms update
        self.S_Wf = (self.beta1 * self._S_Wf + (1 - self.beta1) * grads["dWf"] ** 2) / scal2
        self._S_Wi = (self.beta1 * self._S_Wi + (1 - self.beta1) * grads["dWi"] ** 2) / scal2
        self._S_Wg = (self.beta1 * self._S_Wg + (1 - self.beta1) * grads["dWg"] ** 2) / scal2
        self._S_Wo = (self.beta1 * self._S_Wo + (1 - self.beta1) * grads["dWo"] ** 2) / scal2
        self._S_Wy = (self.beta1 * self._S_Wy + (1 - self.beta1) * grads["dWy"] ** 2) / scal2

        self._S_bf = (self.beta1 * self._S_bf + (1 - self.beta1) * grads["dbf"] ** 2) / scal2
        self._S_bi = (self.beta1 * self._S_bi + (1 - self.beta1) * grads["dbi"] ** 2) / scal2
        self._S_bg = (self.beta1 * self._S_bg + (1 - self.beta1) * grads["dbg"] ** 2) / scal2
        self._S_bo = (self.beta1 * self._S_bo + (1 - self.beta1) * grads["dbo"] ** 2) / scal2
        self._S_by = (self.beta1 * self._S_by + (1 - self.beta1) * grads["dby"] ** 2) / scal2


### AUXILIARY FUNCTIONS
def load_data(file_paths, lower=False):
    data = str()
    for path in file_paths:
        data += open(path, 'r').read()  # provides a single string
    if lower:
        data = data.lower()

    chars = list(set(data))  # return list of of unique characters used. In our case it is the vocabulary.

    char_to_indx = {ch: i for i, ch in enumerate(chars)}
    indx_to_char = {i: ch for i, ch in enumerate(chars)}

    return data, chars, char_to_indx, indx_to_char


def chunks_input_target(data, seq_length, char_to_indx):
    inputs, targets = [], []
    data_input = data[:-1]
    data_output = data[1:]
    for pointer in range(0, len(data) - 1, seq_length):
        input = [char_to_indx[ch] for ch in data_input[pointer:pointer + seq_length]]
        target = [char_to_indx[ch] for ch in data_output[pointer:pointer + seq_length]]
        inputs.append(input)
        targets.append(target)

    return inputs, targets


def one_hot_encoding(vocab_size, k):
    x = np.zeros((vocab_size, 1))
    x[k] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x )

    return e_x / np.sum(e_x)


def cross_entropy(prediction, target):
    return sum(-np.log(prediction[t][target[t], 0]) for t in range(len(prediction)))


if __name__ == '__main__':
    # LOAD DATA
    file_list = [DANTE_PATH]

    data, chars, char_to_indx, indx_to_char = load_data(file_list, lower=True)
    data_size, vocab_size = len(data), len(chars)
    print(data_size)

    # Define RNN architecture
    lstm = LSTM(vocab_size, HIDDEN_SIZE, vocab_size, seq_length=SEQ_LENGHT)
    optimiser = Adam_opt(vocab_size, HIDDEN_SIZE, vocab_size)
    lstm.compile(cross_entropy, optimiser, char_to_indx, indx_to_char)
    lstm.train(data)

    plt.plot(lstm.history)
    plt.show()

    #print(lstm.texts[-5:])
