import numpy as np
import math as m
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy.matlib
import tensorflow.keras.backend as K


#compute the phi wavelet to approximate the triangular frequency filter
def Hanning(t:np.array, s:int)->float :
    #compute the Hannin window
    #variable time is a time vector
    #s is the width of the window
    return np.where(abs(t) <=0.5*s, 0.5+0.5*np.cos(2*m.pi*t/s), 0)

def phi_n(eta_n:float,w_n:float, t:float) -> float :
    #psi_n is the triangle frequency filter, approximated by phi_n
    #t is time
    #triangles centered on eta_n 
    #full width at half maximum (FWHM) w_n
    sigma_n = 2*m.sqrt(2*m.log(2))/w_n
    
    phi_n = np.exp(-2*m.pi*1j*eta_n*t) * (1/(m.sqrt(2*m.pi)*sigma_n)) * np.exp((-t**2)/2*sigma_n**2)
    #what about the normalisation?
    return phi_n

def compute_filter_bank(N:int,w:int, f_start:int,f_end:int) ->np.array:
    #N is the number of filters
    #w is the width in number of samples
    #fstart is the beginning of the frequencies covered by the filterbank and fends its end
    #duree is the width of the filter (length of the window/sample frequency) in number of samples
    #It seems that w and duree are always the same
    #return filter_bank an array with at each column a filter bank, size(duree,N)
    duree = w
    filter_bank_phi = np.zeros((duree,2*N))
    
    #compute mel-scale:
    f_range = np.arange(f_start,f_end,1)
    mel_range = 2595*np.log10(1+f_range/700)
    #take N equaly spaced frequencies
    idx = np.round(np.linspace(0, len(mel_range) - 1, N)).astype(int)
    eta_vect = mel_range[idx]
    
    #create time vector for hanning and gabor
    #centered on 0 and its length is duree
    t = np.arange(-duree/2,duree/2,1)
    i=0
    for eta_n in eta_vect:
        temp_complex = phi_n(eta_n, w, t)
        filter_bank_phi[:,2*i] = temp_complex.real
        filter_bank_phi[:,2*i+1] = temp_complex.imag
        i=i+1
    return filter_bank_phi
    
#Initializer = tf.keras.initializers.ExampleRandomNormal()
def init_TDmel(shape, dtype=None):
    #initialize with the TD mel filterbank
    #shape is (width of the filter, 1, number of filters)
    weight =  compute_filter_bank(N = int(shape[2]*0.5),w = shape[0], f_start = 64,f_end = 8000)
    return weight.reshape(shape)

def init_Hanning(shape, dtype=None):
    duree = 400
    t = np.arange(-duree/2,duree/2,1)
    weight = Hanning(t, s=shape[0])
    weight = K.pow(np.matlib.repmat(weight, 1, shape[2]),2)

    return tf.cast(tf.reshape(weight,shape),dtype=tf.float32)

