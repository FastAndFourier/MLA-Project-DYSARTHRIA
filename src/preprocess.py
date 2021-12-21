from librosa.feature.spectral import mfcc
import numpy as np
import librosa.feature
import librosa.display
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

FS = 16000
NB_FRAME = 251



def compute_mfsc(set):

    if os.path.isfile('../data/mfsc_'+set+'.npy'):

        mfsc = np.load('../data/mfsc_'+set+'.npy')

        return mfsc

    data = np.load('../data/x_'+set+'.npy')
    n_obs = data.shape[0]
    
    
    win_len = int(.025*FS)
    hop_len = int(.01*FS)

    mfscs = np.zeros((n_obs,64,int(NB_FRAME)))

    for k in tqdm(range(n_obs), desc='processing'):
        data_ = librosa.effects.preemphasis(data[k])
        temp_ = librosa.feature.melspectrogram(y=data_, sr=FS,n_mels=64,win_length=win_len,hop_length=hop_len)
        mfscs[k] = np.abs(temp_)

    np.save('../data/mfsc_'+set+'.npy',mfscs,allow_pickle=True)

    return mfscs


def normalize_mfsc(set,method):

    if os.path.isfile('../data/mfsc_'+set+'.npy'):
        mfsc = np.load('../data/mfsc_'+set+'.npy')
    else:
        mfsc = compute_mfsc(set)

    normalized_feature = np.zeros(mfsc.shape)

    if method == "log" and not os.path.isfile('../data/log_'+set+'.npy'):
        for k in range(mfsc.shape[0]):
            normalized_feature[k] = librosa.amplitude_to_db(mfsc[k],ref=np.max)

        np.save('../data/log_'+set+'.npy',normalized_feature)
        

    elif method == "mvn" and not os.path.isfile('../data/mvn_'+set+'.npy'):
        for k in range(mfsc.shape[0]):
            normalized_feature[k] = (mfsc[k] - mfsc[k].mean())/mfsc[k].std()

        np.save('../data/mvn_'+set+'.npy',normalized_feature)

    elif method == "pcen" and not os.path.isfile('../data/pcen_'+set+'.npy'):
        for k in range(mfsc.shape[0]):
            normalized_feature[k] = librosa.pcen(mfsc[k],power=0.5,gain=0.98,bias=2.0,b=0.5,eps=1e-6)

        np.save('../data/pcen_'+set+'.npy',normalized_feature)



    return normalized_feature


if __name__ == "__main__":

    set = ['test','train','val']
    norma = ["log","mvn","pcen"]

    

    for s in set:
        print("Processing",s)
        mfsc = compute_mfsc(s)
        for n in norma:
            normalize_mfsc(mfsc,s,n)
            print("---------",n,"done")




# def preprocess(data_path,mfsc_path,pcen_path,rewrite=False, compression = True):

#     if os.path.isfile(pcen_path) and not(rewrite):
#         mfscs = np.load(mfsc_path,allow_pickle=True)
#         pcen = np.load(pcen_path,allow_pickle=True)

#         return mfscs, pcen

#     print("Processing data ...")
#     data = np.load(data_path)
#     n_obs = data.shape[0]
    
#     fs = 16000
    
#     win_len = int(.025*fs)
#     hop_len = int(.01*fs)

#     nb_frame = 251

#     mfscs = np.zeros((n_obs,64,int(nb_frame)))
#     pcen = np.zeros((mfscs.shape[0],mfscs[0].shape[0],mfscs[0].shape[1]))

#     for k in tqdm(range(n_obs), desc='processing'):
#         data_ = librosa.effects.preemphasis(data[k])
#         mfscs[k] = librosa.feature.melspectrogram(y=data_, sr=fs,n_mels=64,win_length=win_len,hop_length=hop_len)
#         pcen[k] = librosa.pcen(mfscs[k],power=0.5,gain=0.98,bias=2.0,b=0.5,eps=1e-6)
        
#         if compression:
#             mfscs[k] = librosa.amplitude_to_db(np.abs(mfscs[k]),ref=np.max)

#     np.save(mfsc_path,mfscs,allow_pickle=True)
#     np.save(pcen_path,pcen,allow_pickle=True)

#     print("Processing done!")

#     return mfscs, pcen


