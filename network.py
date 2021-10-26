from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.activations import sigmoid 
from keras.layers import CuDNNLSTM, Dropout
from tensorflow.keras.layers import Flatten


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



# def lstm_model(concatenated_input):
#     classifier = Sequential()
#     classifier.add(CuDNNLSTM(1, input_shape=(concatenated_input.shape), return_sequences=True))
#     classifier.add(Dropout(0.2))
#     return sigmoid(classifier(concatenated_input))