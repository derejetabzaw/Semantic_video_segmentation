import os 
import numpy as np 
from keyframe_selector import keyframe_selector as keyframe
import tensorflow as tf 
from tensorflow.keras.layers import Concatenate
from network import cnn_for_convfeats,cnn_for_csift,cnn_for_mfcc,lstm_model
import pandas as pd


df = pd.read_csv('features.csv')



# color_SIFT_features = np.array(color_SIFT_features)
# MFCC_features = np.array(MFCC_features)
# convolution_features_tensors = np.array(convolution_features_tensors)
# convolution_features_tensors = convolution_features_tensors.reshape(1,convolution_features_tensors.shape[0],convolution_features_tensors.shape[1])
# print ("ConFeats to CNN")

# cnn_output_mfcc = np.array(cnn_for_mfcc(MFCC_features))
# cnn_output_csift = np.array(cnn_for_csift(color_SIFT_features))
# cnn_output_convfeats = np.array(cnn_for_convfeats(convolution_features_tensors))



# input_for_lstm = np.concatenate((cnn_output_mfcc.flatten(),cnn_output_csift.flatten(),cnn_output_convfeats.flatten()),axis = None)
# input_for_lstm = input_for_lstm.reshape(input_for_lstm.shape[0],1)


# output = lstm_model(input_for_lstm)
# print (output)