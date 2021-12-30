import os 
import numpy as np 
from keyframe_selector import keyframe_selector as keyframe
from features import convfeats,CSIFT,MFCC, clean_up
import tensorflow as tf 
from tensorflow.keras.layers import Concatenate
from network import cnn_for_convfeats,cnn_for_csift,cnn_for_mfcc,lstm_model

import pandas as pd

data = []



dataset_directory = (os.getcwd() + '/' + "Dataset").replace("\\","/")
keyframes_directory = (dataset_directory + '/' + 'keyframes')

color_SIFT_features = []
MFCC_features = []
count = 0



for root, dirs,files in os.walk(dataset_directory):
    for filename in files:
        if filename.endswith((".mp4",".flv",".mpg",".avi",".wmv",".mpv")):
            target_video = str(dataset_directory + '/' + filename)
            print ("Selecting Keyframes")
            keyframe(target_video)
            print ("Done Selecting")
            print ("Cleaning Up ...")
            clean_up(keyframes_directory)
            print ("Extracting Features")
            for key_root, key_dirs,keyframes in os.walk(keyframes_directory):
                for images in keyframes:
                    if images.endswith((".jpeg",".jpg",".png")):
                        
                        keyframe_image = (keyframes_directory + '/' + images).replace("\\","/")
                        convolution_features = convfeats(keyframe_image)
                        if count == 0:
                            convolution_features_tensors = convolution_features
                        else:
                            convolution_features_tensors = Concatenate(axis = 0)([convolution_features_tensors,convolution_features])
                        color_SIFT_features.append(CSIFT(keyframe_image,100)[1])
                        count += 1
                for audios in keyframes:
                    if audios.endswith((".wav")):
                        keyframe_audio = str(keyframes_directory + '/' + audios).replace("\\","/")
                        keyframe_mfcc = MFCC(keyframe_audio)
                        MFCC_features.append(keyframe_mfcc)
            data.append((filename,convolution_features_tensors,color_SIFT_features,MFCC_features,count))
            print ("Done Extracting")
            

df = pd.DataFrame(data = data, columns = ['filename','convFeats','CSIFT','MFCC','keyframe_count'])
df.to_csv('features.csv',index=False)




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