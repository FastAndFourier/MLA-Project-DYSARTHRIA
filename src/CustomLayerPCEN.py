import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K

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


  def call(self, energy, prec_moving_mean):

    s = self.s
    eps = self.eps 
    m = (1-s)*prec_moving_mean + s*energy

    outputs = (((energy/((eps+m)**self.alpha))+self.delta)**self.r)-self.delta**self.r

    return outputs