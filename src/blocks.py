import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Input, Lambda, Dropout, BatchNormalization
import tensorflow.keras.backend as K

from time_mfb import init_Hanning, init_TDmel
import librosa

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

class CustomLayerPCEN2(tf.keras.layers.Layer):
    "Custom Layer version 2, will replace V1 when test will be done"


    def __init__(self, **kwargs):

        super(CustomLayerPCEN2, self).__init__(**kwargs)
        #States corresponds to the variables to learn (alpha, r and delta) / default values are 0.98/0.5/2.0
        self.trainable = True

    def build(self, input_shape):

        self.alpha = tf.Variable(initial_value=0.98, dtype='float32', name = 'alpha', trainable = True)
        #r needs to stay inside [0,1] / to be tested without constraint and use abs(r) in call
        self.r = tf.Variable(initial_value=0.5, dtype='float32', name = 'r', constraint = lambda t: tf.clip_by_value(t,0,1), trainable=True)
        self.delta = tf.Variable(initial_value=2.0, dtype='float32', name ='delta', trainable=True)

        #Constants corresponds to the constant values used in the computation (s, eps) / default values are 0.5/1e-6
        self.s = tf.Variable(initial_value=0.5, dtype='float32', name = 's', trainable=False)
        self.eps = tf.Variable(initial_value=1e-6, dtype= 'float32', name = 'eps', trainable =False)
    
    def call(self, data):
        s = self.s
        eps = self.eps
        M = []
        #Loop on each feature (corresponding to each frequency in the mfcc) and each timestep
        for i in range(data.shape[-2]):
            prec_mean, current_mean = 0, 0
            M_row = []
            for j in range(data.shape[-1]):
            #First timestep, there is no previous value, so we set prec_mean to zero
                if j == 0:
                    prec_mean = 0
                else:
                    prec_mean = current_mean
                #Compute the current mean based on the previous mean and the current value
                current_mean = (1-s)*prec_mean + s*data[0, i, j]
                M_row.append(current_mean)
            #Compute the value of the PCEN for each t, f
            M.append(M_row)

        Madde = tf.math.add(M, self.eps)
        #print(Madde)
        Mepow = tf.sign(Madde)*tf.pow(tf.abs(Madde), self.alpha)
        #print(Mepow)
        EonM = tf.math.divide(data, Mepow)
        #print(EonM)
        EonM_delta = tf.math.add(EonM, self.delta)
        #print(EonM_delta)
        deltapow = tf.math.pow(self.delta, abs(self.r))
        #print(deltapow)
        pcen = tf.math.subtract(tf.math.pow(EonM_delta, abs(self.r)), deltapow)
        #print(pcen)
    
        # pcen=librosa.pcen(S=data, sr=16000, gain=self.alpha, bias=self.delta, power = self.r, eps=self.eps, b= self.s)

        return pcen