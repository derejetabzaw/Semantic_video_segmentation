import os 
import cv2
import wave
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids



def convfeats(image):
    image = img_to_array(load_img(image, target_size=(224, 224)))

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # 1st Conv Block

    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(image)
    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 2nd Conv Block

    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 3rd Conv block

    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 4th Conv block

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 5th Conv block

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # Fully connected layers

    x = Flatten()(x)
    
    convFeat_tensors = Dense(units = 4096, activation ='relu')(x)
    return convFeat_tensors




def CSIFT(image,cluster_points = 100):
    image = cv2.imread(image,1)
    sift = cv2.xfeatures2d.SIFT_create()
    (key_points,descriptors) = sift.detectAndCompute(image, None)
    points = [key_points[idx].pt for idx in range(0, len(key_points))]
    if len(points) > cluster_points:
        kmean = KMeans(cluster_points)
        kmean.fit(points)
        clustered_points = kmean.cluster_centers_
        backtorgb = cv2.cvtColor(descriptors,cv2.COLOR_GRAY2RGB)
        descriptor_in_rgb = backtorgb.reshape(backtorgb.shape[0],backtorgb.shape[1]*backtorgb.shape[2])
        kmedoids = KMedoids(n_clusters=100, random_state=0,metric='euclidean').fit(descriptor_in_rgb)

        kmedoids_selection = kmedoids.cluster_centers_
        return clustered_points,kmedoids_selection
    else:
        return [0],[0]





def MFCC(audio,cluster_points = 500):

    frequency_sampling, audio_signal = wavfile.read(audio)

    features_mfcc = mfcc(audio_signal, frequency_sampling)

    kmean_for_audio = KMeans(cluster_points)

    if features_mfcc.shape[0] > cluster_points:
        kmean_for_audio.fit(features_mfcc)
        return kmean_for_audio.cluster_centers_
    return None 





def clean_up(folder):
    for root, dirs, files in os.walk('Dataset/keyframes'):
        for filename in files:
            if filename.endswith((".jpeg",".jpg",".png")):
                image_file = str(folder + '/' + filename)
                print ("image_files:",image_file)
                image = cv2.imread(image_file,cv2.IMREAD_COLOR)
                print ("image_data:",image)
                if np.mean(image) <  1:
                    os.remove(image_file)
            if filename.endswith((".wav")):
                audio_file = str(folder + '/' + filename)
                with wave.open(audio_file, "rb") as wave_file:
                    sample = wave_file.getnframes()
                    if sample < 16000 * 5:
                        wave_file.close()
                        delete_blank_audio = audio_file.replace("/","\\")
                        os.remove(delete_blank_audio)
