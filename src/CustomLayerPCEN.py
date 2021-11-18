import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.python.framework import tensor_shape

class CustomLayerPCEN(tf.keras.layers.Layer):

  def __init__(self, **kwargs):

    super(CustomLayerPCEN, self).__init__(**kwargs)
    #States corresponds to the variables to learn (alpha, r and delta) / default values are 0.98/0.5/2.0
    self.alpha = tf.Variable(initial_value=[0.98], dtype='float32', name = 'alpha')
    #r needs to stay inside [0,1] / to be tested without constraint and use abs(r) in call
    self.r = tf.Variable(initial_value=[0.5], dtype='float32', name = 'r', constraint = lambda t: tf.clip_by_value(t,0,1))
    self.delta = tf.Variable(initial_value=[2.0], dtype='float32', name ='delta')

    #Constants corresponds to the constant values used in the computation (s, eps) / default values are 0.5/1e-6
    self.s = tf.constant(value=[0.5], dtype='float32', name = 's')
    self.eps = tf.constant(value=[1e-6],dtype= 'float32', name = 'eps')


  def call(self, data):

    s = self.s
    eps = self.eps
    #res = tf.identity(data)
    res = np.zeros([int(data.shape[-2]), int(data.shape[-1])])
    print(data.shape)
    #Loop on each feature (corresponding to each frequency in the mfcc) and each timestep
    for i in range(data.shape[-2]):
        prec_mean, current_mean = 0, 0
        for j in range(data.shape[-1]):
            #First timestep, there is no previous value, so we set prec_mean to zero
            if j == 0:
                prec_mean = 0
            else:
                prec_mean = current_mean
            #Compute the current mean based on the previous mean and the current value
            current_mean = (1-s)*prec_mean + s*data[i,j]
            #Compute the value of the PCEN for each t, f
            
            res[i,j] = (((data[i,j]/((eps+current_mean)**self.alpha))+self.delta)**self.r)-self.delta**self.r

    return res

if __name__ == '__main__':

    data = np.random.uniform(1,50, size=(20,10))

    test_model = CustomLayerPCEN()
    
    #output_model = test_model(data)
    input = tf.keras.layers.Input(shape=data.shape)
    model = Model(input, test_model(input) )
    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', run_eagerly=True)
    history = model.fit(data, epochs=3, verbose = 1)
    print(model.layers[-1].weights)
    #model.summary()