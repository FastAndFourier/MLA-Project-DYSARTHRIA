from librosa.feature.spectral import mfcc
import numpy as np
import librosa.feature
import librosa.display
import librosa
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import attention_model
import tensorflow as tf
from preprocess import *
import os


if __name__ == "__main__":

    


    # mfscs_train = np.load("../data/mfscs_train.npy",allow_pickle=True)
    # pcen_train = np.zeros((mfscs_train.shape[0],mfscs_train[0].shape[0],mfscs_train[0].shape[1]))

    # for k in range(mfscs_train.shape[0]):
    #     mfscs_train[k] = librosa.amplitude_to_db(mfscs_train[k],ref=np.max)
    #     pcen_train[k] = librosa.pcen(mfscs_train[k]*(2**31),power=0.5,gain=0.98,bias=2.0,b=0.5,eps=1e-6)

    # mfscs_test = np.load("../data/mfscs_test.npy",allow_pickle=True)
    # pcen_test = np.zeros((mfscs_test.shape[0],mfscs_test[0].shape[0],mfscs_test[0].shape[1]))

    # for k in range(mfscs_test.shape[0]):
    #     mfscs_test[k] = librosa.amplitude_to_db(mfscs_test[k],ref=np.max)
    #     pcen_test[k] = librosa.pcen(mfscs_test[k]*(2**31),power=0.5,gain=0.98,bias=2.0,b=0.5,eps=1e-6)

    rewrite = False

    print("--------- Training data ---------\n")
    mfscs_train, pcen_train = preprocess("../data/x_train.npy","../data/mfsc_train.npy","../data/pcen_train.npy",rewrite)
    print("\n--------- Test data ---------\n")
    mfscs_test, pcen_test = preprocess("../data/x_test.npy","../data/mfsc_test.npy","../data/pcen_test.npy",rewrite)


    
    pcen_train = pcen_train.transpose(0,2,1)
    pcen_test = pcen_test.transpose(0,2,1)

    mfsc_train = mfscs_train.transpose(0,2,1)
    mfsc_test = mfscs_test.transpose(0,2,1)

    X_train = pcen_train
    X_test = pcen_test

    print(X_train.shape)


    
    test_model = attention_model(X_train)
    
    test_model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    test_model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=['binary_accuracy'])

    y_train = np.load("../data/y_train.npy")
    y_train = tf.keras.utils.to_categorical(y_train.astype(int),num_classes=2)

    y_test = np.load("../data/y_test.npy")
    y_test = tf.keras.utils.to_categorical(y_test.astype(int),num_classes=2)


    test_model.fit(X_train,y_train,epochs=20,batch_size=64)
    print('accuracy {:.2f}'.format(test_model.evaluate(X_test,y_test)[1]))


    # img = librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='hz', x_axis='time',sr=fs)
    # plt.colorbar(img, format="%+2.0f dB")

    # plt.show()