import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Input, Permute, Multiply, Lambda, Dropout
import tensorflow.keras.backend as K


import argparse

CLASS_NUMBER = 2


def uar_metric(y_true,y_pred):

    pred = np.zeros(y_pred.shape[0])
    true = np.zeros(y_pred.shape[0])

    for k in range(y_pred.shape[0]):
        pred[k] = int(y_pred[k,0,0].numpy() < y_pred[k,0,1].numpy())
        true[k] = int(y_true[k,0].numpy() < y_true[k,1].numpy())

    #print(pred,y_pred.numpy())

    TP = tf.math.count_nonzero(true*pred)
    FP = tf.math.count_nonzero((true - 1) * pred)
    FN = tf.math.count_nonzero((pred - 1) * true)
    TN = np.size(pred) - TP - FP - FN



    return .5*(TP/(TP+FN) + TN/(TN+FP)) #(TP+TN)/(TP+TN+FP+FN) #TN/(TN+FP) # #






class FullModel():

    def __init__(self,args):

        self.frontEnd = args.frontEnd
        self.normalization = args.normalization

        self.model = None

        self.SIZE_INPUT = [64,251]

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.n_ex=args.n_ex


    def build(self):


        if self.frontEnd in ['LLD','melfilt'] :

            input = tf.keras.layers.Input(shape=self.SIZE_INPUT)
            x = input

            if self.normalization == 'learn_pcen':

                x = CustomLayerPCEN2()(input)


            model = Model(input,AttentionModelLayer()(x))

            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr,momentum=0.98)
            #tf.keras.optimizers.Adam(learning_rate=self.lr)
            
            model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics = [uar_metric,'binary_accuracy'],run_eagerly=True)
            model.summary()

            self.model = model

            return model
        
        else:

            return None

    def train(self):

        if self.model == None:
            print("Model has not been build. Please run buildModel")
            return None
        
        if self.frontEnd == 'melfilt':
            if self.normalization == "learn_pcen":
                X_train = np.load('../data/mfsc_train.npy',allow_pickle=True)    
            else:
                X_train = np.load('../data/'+self.normalization+'_train.npy',allow_pickle=True)
        print(X_train.shape)

        y_train = np.load("../data/y_train.npy")

        p = np.random.permutation(len(y_train))

        X_train = X_train[p]
        y_train = y_train[p]

        if self.n_ex==0:
            val=int(input("Number of examples ? "))
            X_train = X_train[0:val]
        
        

        if self.n_ex==0:
            y_train = y_train[0:val]
      
    
        print(len(y_train[y_train==0]),len(y_train[y_train==1]))
        y_train = tf.keras.utils.to_categorical(y_train.astype(int),num_classes=2)


        self.model.fit(X_train, y_train, batch_size = self.batch_size, 
                       epochs = self.epochs, verbose = 1)

        return self.model.history



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



        

class AttentionModelLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionModelLayer, self).__init__()
        self.lstm = LSTM(60, return_sequences = True)
        self.ave_dropout1 = Dropout(rate=0.7)
        self.ave_dense1 = Dense(50)
        self.ave_dropout2 = Dropout(rate=0.7)
        self.ave_dense2 = Dense(1)
        self.ave_softmax = Softmax()
        self.ave_permute = Permute([2,1])
        #self.output_lambda = Lambda(lambda layer: K.sum(layer, axis = -1))
        self.mutliply = Multiply()
        self.layer_output = Dense(CLASS_NUMBER)
        
    def __call__(self, inputs):
        x = self.lstm(inputs)
        #ave = Attention Vector Estimation
        ave = self.ave_dropout1(x)
        ave = self.ave_dense1(x)
        ave = self.ave_dropout2(x)
        ave = self.ave_dense2(ave)
        ave = self.ave_softmax(ave)
        ave = self.ave_permute(ave)
        
        output_attention =  Lambda(lambda layer: K.sum(layer, axis = -1))(x)
        output_attention = self.mutliply([output_attention, ave])
        
        return self.layer_output(output_attention)
    

if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
	

    parser.add_argument('-frontEnd',choices=["LLD","melfilt","TDfilt"],nargs='?',type=str,default="melfilt")
    parser.add_argument('-normalization',choices=["log","mvn","pcen","learn_pcen"],nargs='?',type=str,default="log")
    parser.add_argument('-lr',nargs='?',type=float,default=0.0001)
    parser.add_argument('-batch_size',nargs='?',type=int,default=32)
    parser.add_argument('-epochs',nargs='?',type=int,default=5)
    parser.add_argument('-n_ex', nargs='?', type=int, default=1)
    
    args = parser.parse_args()



    model1 = FullModel(args)
    model1.build()
    model1.train()


    # data = np.random.uniform(1,100, size=(2000,500)) 
    # test_model = attention_model(data[np.newaxis, ...])
    
    # test_model.summary()
    
    # attention_layer = AttentionModelLayer()
    # print('test')
    # y = attention_layer(data[np.newaxis, ...])


    