import os 
import numpy as np 
from keyframe_selector import keyframe_selector as keyframe
from features import convfeats,CSIFT,MFCC, clean_up
import tensorflow as tf 
from tensorflow.keras.layers import Concatenate
from network import cnn_for_convfeats,cnn_for_csift,cnn_for_mfcc


dataset_directory = (os.getcwd() + '/' + "Dataset").replace("\\","/")
keyframes_directory = (dataset_directory + '/' + 'keyframes')

color_SIFT_features = []
MFCC_features = []
dummy_tensors = tf.zeros([1,4096],tf.float32)
count = 0



for root, dirs,files in os.walk(dataset_directory):
    for filename in files:
        if filename.endswith((".mp4",".flv",".mpg",".avi",".wmv",".mpv")):
            target_video = str(dataset_directory + '/' + filename)
            print ("Selecting Keyframes")
            # keyframe(target_video)
            print ("Done Selecting")
            print ("Cleaning Up ...")
            # clean_up(keyframes_directory)
            print ("Extracting Features")
            for key_root, key_dirs,keyframes in os.walk(keyframes_directory):
                for images in keyframes:
                    if images.endswith((".jpeg",".jpg",".png")):
                        
                        # keyframe_image = (keyframes_directory + '/' + images).replace("\\","/")
                        # convolution_features = convfeats(keyframe_image)
                        # if count == 0:
                        #     convolution_features_tensors = convolution_features
                        # else:
                        #     convolution_features_tensors = Concatenate(axis = 0)([convolution_features_tensors,convolution_features])
                        # print (convolution_features_tensors.shape)
                        # cnn_outout_convfeats = cnn_for_convfeats(convolution_features_tensors)
                        # print (cnn_outout_convfeats)
                        # color_SIFT_features.append(CSIFT(keyframe_image,100)[1])
                        count += 1
                for audios in keyframes:
                    if audios.endswith((".wav")):
                        keyframe_audio = str(keyframes_directory + '/' + audios).replace("\\","/")
                        MFCC_features.append(MFCC(keyframe_audio))
            print ("Done Extracting")
            



color_SIFT_features = np.array(color_SIFT_features)
MFCC_features = np.array(MFCC_features)
print ("ConFeats to CNN")

print (MFCC_features.shape)

print (cnn_for_mfcc(MFCC_features))

