import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Input, Permute, Multiply, Lambda
import tensorflow.keras.backend as K

CLASS_NUMBER = 2

def attention_model(inputs):
    input_layer = Input(shape=inputs.shape[1:])
    x = LSTM(60, return_sequences=True)(input_layer)
    #ave = Attention Vector Estimation
    ave = Dense(50)(x)
    ave = Dense(1)(ave)
    ave = Softmax()(ave)
    ave = Permute([2, 1])(ave)
    
    output_attention =  Lambda(lambda layer: K.sum(layer, axis = -1))(x)
    output_attention = Multiply()([output_attention, ave])
    
    output = Dense(CLASS_NUMBER)(output_attention)
    return Model(input_layer,output)

class AttentionModelLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionModelLayer, self).__init__()
        self.lstm = LSTM(60, return_sequences = True)
        self.ave_dense1 = Dense(50)
        self.ave_dense2 = Dense(1)
        self.ave_softmax = Softmax()
        self.ave_permute = Permute([2,1])
        #self.output_lambda = Lambda(lambda layer: K.sum(layer, axis = -1))
        self.mutliply = Multiply()
        self.layer_output = Dense(CLASS_NUMBER)
        
    def __call__(self, inputs):
        x = self.lstm(inputs)
        #ave = Attention Vector Estimation
        ave = self.ave_dense1(x)
        ave = self.ave_dense2(ave)
        ave = self.ave_softmax(ave)
        ave = self.ave_permute(ave)
        
        output_attention =  Lambda(lambda layer: K.sum(layer, axis = -1))(x)
        output_attention = self.mutliply([output_attention, ave])
        
        return self.layer_output(output_attention)
    

if __name__ == '__main__':
    data = np.random.uniform(1,100, size=(2000,500)) 
    test_model = attention_model(data[np.newaxis, ...])
    
    test_model.summary()
    
    attention_layer = AttentionModelLayer()
    print('test')
    y = attention_layer(data[np.newaxis, ...])


    