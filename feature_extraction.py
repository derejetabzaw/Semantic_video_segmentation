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
                            convolution_features_tensors = np.array(Concatenate(axis = 0)([convolution_features_tensors,convolution_features]))
                        if len(CSIFT(keyframe_image,100)[1]) == 1:
                            os.remove(keyframe_image)
                        else:       
                            color_SIFT_features.append(np.array(CSIFT(keyframe_image,100)[1])) 
                        

                        count += 1
                for audios in keyframes:
                    if audios.endswith((".wav")):
                        keyframe_audio = str(keyframes_directory + '/' + audios).replace("\\","/")
                        keyframe_mfcc = MFCC(keyframe_audio)
                        MFCC_features.append(np.array(keyframe_mfcc))
            data.append((filename,convolution_features_tensors,color_SIFT_features,MFCC_features,count))
            print ("Done Extracting")
            
df = pd.DataFrame(data = data, columns = ['filename','convFeats','CSIFT','MFCC','keyframe_count'])
df.to_csv('features.csv',index=False)