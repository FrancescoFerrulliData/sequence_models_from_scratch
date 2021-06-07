import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# CONSTANTS
DANTE_PATH = 'dante.txt'
SHAK_PATH = 'shakespeare.txt'
FILEPATH_1 = 'deamicis.txt'
FILEPATH_2 = 'deamicis2.txt'

HIDDEN_SIZE = 85
SEQ_LENGHT = 30
EPOCHS = 35
LEARNING_RATE = 1e-2


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


def one_hot_encoding(vocab_size, k):
    x = np.zeros((vocab_size, 1))
    x[k] = 1
    return x


def chunks_input_target(data, seq_length, char_to_indx, vocab_size):
    inputs, targets = [], []
    data_input = data[:-1]
    data_output = data[1:]
    for pointer in range(0, len(data) - 1, seq_length):
        input = [one_hot_encoding(vocab_size, char_to_indx[ch]) for ch in data_input[pointer:pointer + seq_length]]
        target = [one_hot_encoding(vocab_size, char_to_indx[ch]) for ch in data_output[pointer:pointer + seq_length]]
        inputs.append(input)
        targets.append(target)
    if len(inputs[-1]) != seq_length:
        inputs, targets = inputs[:-1], targets[:-1]

    inputs = np.reshape(inputs, (len(inputs), len(inputs[0]), len(inputs[0][0])))
    targets = np.reshape(targets, (len(targets), len(targets[0]), len(targets[0][0])))

    return inputs, targets


def prepare_dataset(file_list, test_size, validation_size):
    data, chars, char_to_indx, indx_to_char = load_data(file_list, lower=True)

    inputs, targets = chunks_input_target(data, SEQ_LENGHT, char_to_indx, len(chars))

    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, len(chars)


def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(HIDDEN_SIZE, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True))

    model.add(keras.layers.TimeDistributed(keras.layers.Dense(input_shape[1])))
    model.add(keras.layers.Activation('softmax'))
    return model


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == '__main__':
    # LOAD DATA
    file_list = [FILEPATH_1, FILEPATH_2]

    X_train, X_validation, X_test, y_train, y_validation, y_test, vocab_size = prepare_dataset(file_list, 0.2, 0.2)
    print(X_train.shape)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=EPOCHS)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
