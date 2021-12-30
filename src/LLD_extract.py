from audinterface.core.feature import Feature
import opensmile
import sox
import librosa
import soundfile
import os
import numpy as np

from tqdm import tqdm

#How to install sox : 
# 1)Download sox-14.4.2-win32.exe from : https://sourceforge.net/projects/sox/files/sox/14.4.2/
# 2) Install it
# 3) Add to the environment variables the path to sox, in C:\Program Files (x86)\sox-14-4-2 for example
# 4) Enjoy and smile openly :)

#How to load config:
#1) change the .conf file in config folder by adding on beginning :
#;;; source
#\{\cm[source{?}:include external source]}

#and at the end
#;;; sink
#\{\cm[sink{?}:include external sink]}

#2) Comment the lines : ;\{../shared/standard_wave_input.conf.inc} and  ;\{../shared/standard_data_output.conf.inc}


def extract_LLD(data, config = 'IS09'):

    path_config ="../../opensmile/config/is09-13/"
    LLD_n, LLD_deltas = [], []

    #Load the config for Interspeech 2009
    #Define the features extractors (LLD and LLD-delta):
    smile_lld = opensmile.Smile(
    feature_set=path_config+'IS09_emotion.conf',
    feature_level='lld',
    )

    smile_lld_de = opensmile.Smile(
    feature_set=path_config+'IS09_emotion.conf',
    feature_level='lld_de',
    # loglevel=2,
    # logfile='smile.log',
    )

    for i in tqdm(range(0, data.shape[0])):
        LLD_n.append(smile_lld.process_signal(
        signal= data[i],
        sampling_rate= 16000
        ))
        #print(LLD_n)
        LLD_deltas.append(smile_lld_de.process_signal(
        signal= data[i],
        sampling_rate= 16000
        ))
        #print(LLD_n, LLD_deltas)
        #input()

    return LLD_n, LLD_deltas

def compute_LLD_files(path, lld_name, lld_d_name):
    """This function loads the lld and lld_deltas arrays, and concatenate them into one single matrix.
    It includes zero padding for the lld part because there are size(lld)=size(lld_deltas)-2
    (One is missing at the beginning and at the end)
    Stack the lld first and then lld_deltas for each example and each window"""

    y1 = np.load(path+lld_name)
    y2= np.load(path+lld_d_name)
    print(y1.shape, y2.shape)
    pad = np.zeros(shape=(16))

    y3 = np.zeros(shape=(y2.shape[0], y2.shape[1], 2*y2.shape[2]))

    for i in range(y2.shape[0]):
        for j in range(y2.shape[1]):
            if(j==0 or j==(y2.shape[1])-1):
                # y3[i,j,:16]=pad
                # y3[i,j,16:]=y2[i,j,:]
                y3[i,j,:]=np.concatenate((pad, y2[i,j,:]), axis = 0)
                # print("padding", y3[i,j])
                # input()
            else:
                # y3[i,j,:16]=y1[i,j-1,:]
                # y3[i,j,16:]=y2[i,j,:]
                y3[i,j,:]=np.concatenate((y1[i,j-1,:], y2[i,j,:]), axis = 0)
                # print(y3[i,j])
                # input()
    print(y3.shape)
    return y3

path="../data/"
path_config ="../../opensmile/config/is09-13/"

#==================================================================#

#Compute the lld and lld_d for each dataset

#Load arrays containing the audio data
# x_test = np.load(path+"x_test.npy")
# x_train = np.load(path+"x_train.npy")
# x_val = np.load(path+"x_val.npy")

# #Extract LLD and LLD_deltas and save to .npy
# lld_test, lld_d_test = extract_LLD(x_test, config = 'IS09')
# np.save(path+"/lld_is09_test", np.array(lld_test))
# np.save(path+"/lld_d_is09_test", np.array(lld_d_test))
# print("Test Done")

# lld_train, lld_d_train = extract_LLD(x_train, config = 'IS09')
# np.save(path+"/lld_is09_train", np.array(lld_train))
# np.save(path+"/lld_d_is09_train", np.array(lld_d_train))
# print("Train Done")

# lld_val, lld_d_val = extract_LLD(x_val, config = 'IS09')
# np.save(path+"/lld_is09_val", np.array(lld_val))
# np.save(path+"/lld_d_is09_val", np.array(lld_d_val))
# print("Val Done")


#================================================================================#

#Concatenate the lld and lld_d dats into one element for each dataset
names = [["lld_is09_train.npy", "lld_d_is09_train.npy"], ["lld_is09_test.npy", "lld_d_is09_test.npy"], ["lld_is09_val.npy", "lld_d_is09_val.npy"]]

# for i in range(len(names)) :
#     data=compute_LLD_files(path, names[i][0], names[i][1])
#     if(i==0):
#         np.save(path+"/full_lld_is09_train", np.array(data))
#     if(i==1):
#         np.save(path+"/full_lld_is09_test", np.array(data))
#     if(i==2):
#         np.save(path+"/full_lld_is09_val", np.array(data))


#==============To verifiy==============
# y_f=np.load(path+"/full_lld_is09_val.npy")
# y_1=np.load(path+"/lld_is09_val.npy")
# y_2=np.load(path+"/lld_d_is09_val.npy")

# print(y_f.shape, y_1.shape, y_2.shape)

# print(y_f[0,249,:])
# print(y_1[0,248,:])
# print(y_2[0,249,:])

