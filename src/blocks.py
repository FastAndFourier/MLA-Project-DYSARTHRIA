
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Input, Lambda, Dropout, BatchNormalization
import tensorflow.keras.backend as K

from time_mfb import init_Hanning, init_TDmel





def TD_filt(x):

    x = tf.keras.layers.Conv1D(filters = 128, kernel_size = 400, activation='relu',
                               bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_TDmel)(x)
    #x = Dropout(rate=0.5)(x)
    #compute L2 Norm 
    a = x[:,:,::2]#even elements (real part)
    b = x[:,:,1::2]#uneven elements (imaginary part)
    y = K.pow(a,2)+K.pow(b,2)#norm of elements (only 40 channels now)

    #apply hanning window separately on each 40 channels (need to repmat hanning and do grouped conv)
    y = tf.keras.layers.Conv1D(filters = 64, kernel_size = 400, groups = 64, strides = 160, activation='relu',
                            bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_Hanning)(y)
    #y = Dropout(rate=0.5)(y)
    y = K.abs(y)
    y = K.log(1+y)

    return y

def Attention(x):

    
    y = Dense(50,kernel_regularizer=tf.keras.regularizers.l2(0.001),
              kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None))(x)
    
    y = Dropout(rate=0.5)(y)
    y = BatchNormalization()(y)    
    
    y = Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.001),
              kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None))(y)
    y = Dropout(rate=0.5)(y)
    y = BatchNormalization()(y) 
    
    y = Lambda(lambda x : K.squeeze(x,axis=-1))(y)
    y = Softmax(axis=-1)(y)
    
    
    return y

