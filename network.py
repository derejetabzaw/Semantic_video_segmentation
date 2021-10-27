from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import LSTM
from tensorflow.keras.layers import Flatten, Dense


def cnn_for_mfcc(input):
    model = models.Sequential()
    model.add(layers.Conv1D(32, 1, activation='relu', batch_input_shape=input.shape))
    model.add(layers.MaxPooling1D((6)))
    model.add(layers.Conv1D(64, (1), activation='relu'))
    model.add(layers.MaxPooling1D((7)))
    model.add(layers.Conv1D(128, (1), activation='relu'))
    model.add(layers.MaxPooling1D((8)))
    model.add(layers.Flatten())
    model.summary()
    return model(input)

def cnn_for_convfeats(input):
    model = models.Sequential()
    model.add(layers.Conv1D(1024, 1, activation = 'relu', input_shape=input.shape[1:]))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(256, 1,activation = 'relu'))
    model.add(layers.MaxPooling1D(2))

    model.add(Flatten())
    model.summary()
    return model(input)


def cnn_for_csift(input):
    model = models.Sequential()
    model.add(layers.Conv1D(300, 1, activation='relu', batch_input_shape=input.shape))
    model.add(layers.MaxPooling1D(4))

    model.add(layers.Conv1D(256, 1,activation = 'relu'))
    model.add(layers.MaxPooling1D(4))

    model.add(layers.Conv1D(128, 1 , activation = 'relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(Flatten())
    model.summary()
    return model(input)

def lstm_model(input):
    model = Sequential()
    model.add(LSTM(500, input_shape=input.shape, return_sequences = True))
    model.add(LSTM(500, return_sequences = True))
    model.add(LSTM(500))
    model.add(Dense(1,activation = 'sigmoid'))
    model.summary()
    input = input.reshape(1,input.shape[0],input.shape[1])
    return model(input)

