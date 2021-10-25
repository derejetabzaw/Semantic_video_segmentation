from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.activations import sigmoid 
from keras.layers import CuDNNLSTM, Dropout
from tensorflow.keras.layers import Flatten

def cnn_for_mfcc(input):
    model = models.Sequential()
    model.add(layers.Conv2D(13, 32, activation = 'relu', input_shape=input.shape))
    model.add(layers.MaxPooling2D(6))

    model.add(layers.Conv2D(32, 64,activation = 'relu'))
    model.add(layers.MaxPooling2D(7))

    model.add(layers.Conv2D(64, 128 , activation = 'relu'))
    model.add(layers.MaxPooling2D(8))
    model.add(Flatten())
    model.summary()
    return model(input)


def cnn_for_convfeats(input):
    model = models.Sequential()
    model.add(layers.Conv1D(4096, 1024, activation = 'relu', input_shape=input.shape))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(1024, 256,activation = 'relu'))
    model.add(layers.MaxPooling1D(2))

    model.add(Flatten())
    return model(input)


def cnn_for_csift(input):
    model = models.Sequential()
    model.add(layers.Conv1D(384, 300, activation = 'relu', input_shape=input.shape))
    model.add(layers.MaxPooling1D(4))

    model.add(layers.Conv1D(300, 256,activation = 'relu'))
    model.add(layers.MaxPooling1D(4))

    model.add(layers.Conv1D(256, 128 , activation = 'relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(Flatten())
    return model(input)



# def lstm_model(concatenated_input):
#     classifier = Sequential()
#     classifier.add(CuDNNLSTM(1, input_shape=(concatenated_input.shape), return_sequences=True))
#     classifier.add(Dropout(0.2))
#     return sigmoid(classifier(concatenated_input))