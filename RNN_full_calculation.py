import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
DANTE_PATH = 'dante.txt'
FILEPATH_1 = 'deamicis.txt'  # should be simple plain text file
FILEPATH_2 = 'deamicis2.txt'

HIDDEN_SIZE = 400  # size of hidden layer of neurons
SEQ_LENGHT = 20  # number of steps to unroll the RNN for. Equivalent concept of mini batch.
LEARNING_RATE = 1e-2


class RNN(object):

    def __init__(self, input_size, hidden_size, ouput_size, seq_length=SEQ_LENGHT, activation_fn=np.tanh,
                 learning_rate=LEARNING_RATE):
        self.input_size = input_size
        self.ouput_size = ouput_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self._hidden_state = np.zeros((self.hidden_size, 1))

        self.Wxh, self.Whh, self.Why, self.bh, self.by = set_model_parameters(input_size, hidden_size, ouput_size)
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
        self.history = [-np.log(1.0 / self.input_size) * self.seq_length]
        self.texts = []

    def compile(self, loss, optimiser, char_to_indx, indx_to_char):
        self.loss = loss
        self.optimiser = optimiser
        self.encoding = char_to_indx
        self.decoding = indx_to_char

    def forward(self, input, hidden_previous):
        """
        inputs is list of integers.
        hidden_previous is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        # Empty dictionaries of the type dict[time_step] = value_at_time_t
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hidden_previous)  # computation of hidden state at time t=0 requires previous value (t=-1)

        for t in range(len(input)):
            xs[t] = one_hot_encoding(self.input_size, input[t])
            hs[t] = self.activation_fn(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # vector of characters' probabilities happening at next step

        return xs, hs, ps

    def store_last_internal_cell_status(self, hs):
        self._hidden_state = hs[len(hs)-2]

    def backward(self, xs, hs, ps, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(ps))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y.
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        return dWxh, dWhh, dWhy, dbh, dby

    def update_weights(self, dWxh, dWhh, dWhy, dbh, dby):
        self.optimiser.update_memory_weigths(dWxh, dWhh, dWhy, dbh, dby)
        for w, dw, mw in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby],
                             [self.optimiser._Adag_Wxh, self.optimiser._Adag_Whh, self.optimiser._Adag_Why,
                              self.optimiser._Adag_bh, self.optimiser._Adag_by]):
            w += -self.learning_rate * dw / np.sqrt(mw + 1e-8)

    def train(self, data):
        # loss at iteration 0: the probability of having n=seq_length randomly chosen character from a dictionary
        # containing i=input_size number of different characters.

        inputs, targets = chunks_input_target(data, self.seq_length, self.encoding)
        for iteration, (input, target) in enumerate(zip(inputs, targets)):
            # forward
            xs, hs, ps = self.forward(input, self._hidden_state)
            self.store_last_internal_cell_status(hs)

            # compute time-batch-loss
            iteration_loss = self.loss(ps, target)

            # backward
            dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, target)

            # update model
            self.update_weights(dWxh, dWhh, dWhy, dbh, dby)

            # compute total loss
            smooth_loss = self.history[-1] * 0.999 + iteration_loss * 0.001

            if iteration % 100 == 0:
                print('iter {}, loss: {}'.format(iteration, smooth_loss))
                self.texts.append(self.sample(inputs[0], 8, self.decoding))

            self.history.append(smooth_loss)

        #return history, texts

    def sample(self, seed_ix, n, indx_to_char):

        h = self._hidden_state
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        txt = str()
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            txt += indx_to_char[ix]
        print(txt)
        return txt


class Ada_grad(object):

    def __init__(self, input_size, hidden_size, ouput_size):
        # memory weights
        self._Adag_Wxh, self._Adag_Whh, self._Adag_Why, self._Adag_bh, self._Adag_by = set_adagrad_variables(input_size,
                                                                                                             hidden_size,
                                                                                                             ouput_size)

    def update_memory_weigths(self, dWxh, dWhh, dWhy, dbh, dby):
        self._Adag_Wxh += dWxh * dWxh
        self._Adag_Whh += dWhh * dWhh
        self._Adag_Why += dWhy * dWhy
        self._Adag_bh += dbh * dbh
        self._Adag_by += dby * dby


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


def set_model_parameters(input_size, hidden_size, ouput_size):
    # weights
    Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    Why = np.random.randn(ouput_size, hidden_size) * 0.01  # hidden to output
    # biases
    bh = np.zeros((hidden_size, 1))  # hidden bias
    by = np.zeros((ouput_size, 1))  # output bias

    return Wxh, Whh, Why, bh, by


def set_adagrad_variables(input_size, hidden_size, ouput_size):
    Adag_Wxh = np.zeros((hidden_size, input_size))
    Adag_Whh = np.zeros((hidden_size, hidden_size))
    Adag_Why = np.zeros((ouput_size, hidden_size))
    Adag_bh = np.zeros((hidden_size, 1))
    Adag_by = np.zeros((ouput_size, 1))

    return Adag_Wxh, Adag_Whh, Adag_Why, Adag_bh, Adag_by


def cross_entropy(prediction, target):
    return sum(-np.log(prediction[t][target[t], 0]) for t in range(len(prediction)))


if __name__ == '__main__':
    # LOAD DATA
    file_list = [FILEPATH_1, DANTE_PATH, FILEPATH_2]

    data, chars, char_to_indx, indx_to_char = load_data(file_list)
    data_size, vocab_size = len(data), len(chars)
    print(data_size)

    # Define RNN architecture
    rnn = RNN(vocab_size, HIDDEN_SIZE, vocab_size, seq_length=SEQ_LENGHT)
    optimiser = Ada_grad(vocab_size, HIDDEN_SIZE, vocab_size)
    rnn.compile(cross_entropy, optimiser, char_to_indx, indx_to_char)

    rnn.train(data)
    plt.plot(rnn.history)
    plt.show()

    print(rnn.texts[-5:])
