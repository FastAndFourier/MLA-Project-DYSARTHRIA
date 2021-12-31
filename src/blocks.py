
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
    y = Lambda(lambda a,b : K.pow(a,2)+K.pow(b,2))([a,b])#norm of elements (only 40 channels now)

    #apply hanning window separately on each 40 channels (need to repmat hanning and do grouped conv)
    y = tf.keras.layers.Conv1D(filters = 64, kernel_size = 400, groups = 64, strides = 160, activation='relu',
                            bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_Hanning)(y)
    #y = Dropout(rate=0.5)(y)
    y = Lambda(lambda x : K.abs(x))(y)
    y = Lambda(lambda x : K.log(1+x))(x)

    return y

def Attention(x):

    
    y = Dense(50,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    y = Dropout(rate=0.5)(y)
    y = BatchNormalization()(y)    
    
    y = Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)
    y = Dropout(rate=0.5)(y)
    y = BatchNormalization()(y) 
    
    y = Lambda(lambda x : K.squeeze(x,axis=-1))(y)
    y = Softmax(axis=-1)(y)
    
    
    return y

