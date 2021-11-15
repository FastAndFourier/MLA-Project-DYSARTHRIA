import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Input, Permute, Multiply, Lambda
import tensorflow.keras.backend as K

CLASS_NUMBER = 2

def attention_model(inputs):
    input_layer = Input(shape=inputs.shape)
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


if __name__ == '__main__':
    data = np.random.uniform(1,100, size=(2000,500)) 
    test_model = attention_model(data.reshape((2000,500)))
    
    test_model.summary()