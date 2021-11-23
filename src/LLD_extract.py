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

path="../data/"
path_config ="../../opensmile/config/is09-13/"

#Load arrays containing the audio data
x_test = np.load(path+"x_test.npy")
x_train = np.load(path+"x_train.npy")
x_val = np.load(path+"x_val.npy")

#Extract LLD and LLD_deltas and save to .npy
lld_test, lld_d_test = extract_LLD(x_test, config = 'IS09')
np.save(path+"/lld_is09_test", np.array(lld_test))
np.save(path+"/lld_d_is09_test", np.array(lld_d_test))
print("Test Done")

lld_train, lld_d_train = extract_LLD(x_train, config = 'IS09')
np.save(path+"/lld_is09_train", np.array(lld_train))
np.save(path+"/lld_d_is09_train", np.array(lld_d_train))
print("Train Done")

lld_val, lld_d_val = extract_LLD(x_val, config = 'IS09')
np.save(path+"/lld_is09_val", np.array(lld_val))
np.save(path+"/lld_d_is09_val", np.array(lld_d_val))
print("Val Done")

