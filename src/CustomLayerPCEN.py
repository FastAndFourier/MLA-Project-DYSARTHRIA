import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess
from model import AttentionModelLayer

class CustomLayerPCEN(tf.keras.layers.Layer):

  def __init__(self, **kwargs):

    super(CustomLayerPCEN, self).__init__(**kwargs)
    #States corresponds to the variables to learn (alpha, r and delta) / default values are 0.98/0.5/2.0
    self.alpha = tf.Variable(initial_value=0.98, dtype='float32', name = 'alpha')
    #r needs to stay inside [0,1] / to be tested without constraint and use abs(r) in call
    self.r = tf.Variable(initial_value=0.5, dtype='float32', name = 'r', constraint = lambda t: tf.clip_by_value(t,0,1))
    self.delta = tf.Variable(initial_value=2.0, dtype='float32', name ='delta')

    #Constants corresponds to the constant values used in the computation (s, eps) / default values are 0.5/1e-6
    self.s = tf.constant(value=0.5, dtype='float32', name = 's')
    self.eps = tf.constant(value=1e-6,dtype= 'float32', name = 'eps')


  def call(self, data):

    s = self.s
    eps = self.eps
    #res = tf.identity(data)
    res = data.numpy()#np.zeros([int(data.shape[-2]), int(data.shape[-1])])
    print(data.shape)
    print(type(data))
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
  
class CustomLayerPCEN2(tf.keras.layers.Layer):
  "Custom Layer version 2, will replace V1 when test will be done"
  
  
  def __init__(self, **kwargs):

    super(CustomLayerPCEN2, self).__init__(**kwargs)
    #States corresponds to the variables to learn (alpha, r and delta) / default values are 0.98/0.5/2.0
    self.alpha = tf.Variable(initial_value=0.98, dtype='float32', name = 'alpha')
    #r needs to stay inside [0,1] / to be tested without constraint and use abs(r) in call
    self.r = tf.Variable(initial_value=0.5, dtype='float32', name = 'r', constraint = lambda t: tf.clip_by_value(t,0,1))
    self.delta = tf.Variable(initial_value=2.0, dtype='float32', name ='delta')

    #Constants corresponds to the constant values used in the computation (s, eps) / default values are 0.5/1e-6
    self.s = tf.constant(value=0.5, dtype='float32', name = 's')
    self.eps = tf.constant(value=1e-6,dtype= 'float32', name = 'eps')


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
        current_mean = (1-s)*prec_mean + s*data[0,i,j]
        M_row.append(current_mean)
        #Compute the value of the PCEN for each t, f
      M.append(M_row)
    
    Madde = tf.math.add(M,self.eps)
    Mepow = tf.math.pow(Madde, self.alpha)
    EonM = tf.math.divide(data, Mepow)
    EonM_delta = tf.math.add(EonM, self.delta)
    deltapow = tf.math.pow(self.delta, self.r)
    pcen = tf.math.subtract(tf.math.pow(EonM_delta, self.r), deltapow)
        
    return pcen


if __name__ == '__main__':
  mfsc_train, pcen_train = preprocess('../data/x_train.npy', '../data/mfsc_nocompress_train.npy', '../data/pcen_nocompress_train.npy', compression=False)
  y_train = np.load("../data/y_train.npy")
  y_train = tf.keras.utils.to_categorical(y_train.astype(int),num_classes=2)
  print(mfsc_train.shape)

  data = np.random.uniform(1,50, size=(20,10))
  y = np.random.randint(0,1, size=(20,10))

  test_model = CustomLayerPCEN2()
  attention_layer = AttentionModelLayer()
  opt = Adam(lr=0.001)

  #test = test_model(data)
  #train((mfsc_train,y_train), 2)
  #output_model = test_model(data)
  input = tf.keras.layers.Input(shape=mfsc_train.shape[1:])
  x = test_model(input)
  res = attention_layer(x)
  model = Model(input, res)
  model.summary()
  model.compile(optimizer='adam', loss = 'binary_crossentropy', run_eagerly=True)
  print(data[np.newaxis, ...].shape)
  history = model.fit(mfsc_train, y_train, batch_size = 1, epochs=3, verbose = 1)
  print(model.layers[-1].weights)
  