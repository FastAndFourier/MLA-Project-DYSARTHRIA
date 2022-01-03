import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Input, Permute, Multiply, Lambda, Dropout, Conv2D, BatchNormalization
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from time_mfb import init_Hanning, init_TDmel




from sklearn.metrics import log_loss

import argparse

from tensorflow.python.keras.layers.core import Flatten

from sklearn.model_selection import train_test_split

import os, json 
from time import time

CLASS_NUMBER = 2

N_NEG = 3182/(3182+1382)
N_POS = 1382/(3182+1382)

# def uar_metric(y_true,y_pred):

#     pred = np.zeros(y_pred.shape[0])
#     true = np.zeros(y_pred.shape[0])

#     for k in range(y_pred.shape[0]):
#         pred[k] = int(y_pred[k,0,0].numpy() < y_pred[k,0,1].numpy())
#         true[k] = int(y_true[k,0].numpy() < y_true[k,1].numpy())

#     #print(pred,y_pred.numpy())

#     TP = tf.math.count_nonzero(true*pred)
#     FP = tf.math.count_nonzero((true - 1) * pred)
#     FN = tf.math.count_nonzero((pred - 1) * true)
#     TN = np.size(pred) - TP - FP - FN


#     return .5*(TP/(TP+FN) + TN/(TN+FP))


# Code found on : https://stackoverflow.com/questions/54285037/how-can-i-get-accuracy-from-confusion-matrix-in-tensorflow-or-keras-in-the-form
def non_nan_average(x):
    # Computes the average of all elements that are not NaN in a rank 1 tensor
    nan_mask = tf.math.is_nan(x)
    x = tf.boolean_mask(x, tf.logical_not(nan_mask))
    return K.mean(x)



def uar_metric(y_true, y_pred):
    
    pred_class_label = K.argmax(y_pred, axis=-1)
    true_class_label = K.argmax(y_true, axis=-1)

    cf_mat = tf.math.confusion_matrix(true_class_label, pred_class_label )

    diag = tf.linalg.tensor_diag_part(cf_mat)   

    # Calculate the total number of data examples for each class
    total_per_class = tf.reduce_sum(cf_mat, axis=1)

    acc_per_class = diag / tf.maximum(1, total_per_class)  
    uar = non_nan_average(acc_per_class)

    return uar

def weighted_binary_crossentropy(y_true,y_pred):

    true_label = float(K.argmax(y_true, axis=-1))

    return tf.math.reduce_sum(-tf.math.add(N_POS*true_label*tf.math.log(y_pred[:,:,1]),N_NEG*(1-true_label)*tf.math.log(y_pred[:,:,0])))/len(y_true) 







class FullModel():

    def __init__(self,args):

        self.frontEnd = args.frontEnd
        self.backEnd = args.backEnd
        self.normalization = args.normalization
        self.loss = args.loss

        self.model = None

        self.SIZE_INPUT = [64,251]

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.n_ex=args.n_ex


    def build(self,optimizer="sgd"):


        if self.frontEnd in ['LLD','melfilt','TDfilt'] :

            if self.frontEnd == 'TDfilt':
                input = tf.keras.layers.Input(shape=[40000,1])#size of the raw signal in time domain
                x = TD_melfilter()(input)

            if self.frontEnd == 'LLD':
                input = tf.keras.layers.Input(shape=[251, 32])# Number of windows and number of lld + lld_deltas (2*16)
                x = input
            else:
                input = tf.keras.layers.Input(shape=self.SIZE_INPUT)#,batch_size=self.batch_size)
                x = input
                
            if self.normalization == 'learn_pcen':

                x = CustomLayerPCEN2()(input)


            if self.backEnd == "attention":
                model = Model(input, AttentionModelLayer()(x))
            else:
                model = Model(input, CNNModelLayer()(x))

            if self.loss == "normal_ce":
                loss_ = 'binary_crossentropy'#tf.keras.losses.BinaryCrossentropy
            else:
                loss_ = weighted_binary_crossentropy
                
            # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr,
            #                                                              decay_steps=1000,
            #                                                              decay_rate=0.70,
            #                                                              staircase=True)
            
            # if type(optimizer)!= str:
            #     optimizer.learning_rate = lr_schedule

            optimizer.momentum = 0.98
            model.compile(optimizer=optimizer, loss = loss_, metrics = ['binary_accuracy', uar_metric], run_eagerly=True)
            model.summary()

            self.model = model

            return model
        
        else:

            return None

    def train(self,path="../data/"):

        if self.model == None:
            print("Model has not been build. Please run buildModel")
            return None
        
        if self.frontEnd == 'melfilt':

            if self.normalization in ["none","learn_pcen"]:
                X_train = np.load(path+'mfsc_train.npy',allow_pickle=True) 
                X_val = np.load(path+'mfsc_val.npy',allow_pickle=True)      
            else:
                
                X_train = np.load(path+self.normalization+'_train.npy',allow_pickle=True)
                X_val = np.load(path+self.normalization+'_val.npy',allow_pickle=True)
               
        if self.frontEnd == 'TDfilt':
            X_train = np.load(path+'x_train.npy',allow_pickle=True)
            X_val = np.load(path+'x_val.npy',allow_pickle=True)

        if self.frontEnd == 'LLD':
            X_train = np.load(path+'full_lld_is09_train.npy',allow_pickle=True)
            X_val = np.load(path+'full_lld_is09_val.npy',allow_pickle=True)
           

        y_train = np.load(path+"y_train.npy")
        y_val = np.load(path+"y_val.npy")


#         dataX = np.concatenate((X_train,X_val),axis=0)
#         dataY = np.concatenate((y_train,y_val),axis=0)


#         X_train, X_val, y_train, y_val = train_test_split(dataX, dataY, test_size=1 - 0.80)
       


        if self.n_ex==0:
            val=int(input("Number of examples ? "))
            X_train = X_train[0:val]
        
        

        if self.n_ex==0:
            y_train = y_train[0:val]
      

        y_train = tf.keras.utils.to_categorical(y_train.astype(int),num_classes=2)
        y_val = tf.keras.utils.to_categorical(y_val.astype(int),num_classes=2)


        print("x_shape",X_train.shape)
        self.model.fit(X_train, y_train, batch_size = self.batch_size, 
                       epochs = self.epochs, verbose = 1, validation_data=(X_val,y_val))
        print("ablaee")
        return self.model.history

    
    def save(self,metrics_val,metrics_test):

        if not(os.path.isdir('../model_archive')):
            os.mkdir('../model_archive')

        if not os.path.isfile('../registry.json'):
            data = {}
        else:
            with os.open('../registry.json') as f:
                data = json.load(f)

        id_model = np.random.randint(low=0, high=10**8)

        data[id_model] = {
            "frontEnd" : self.frontEnd,
            "backEnd" : self.backEnd,
            "normalization" : self.normalization,
            "lr" : self.lr,
            "batch_size" : self.batch_size,
            "epochs" : self.epochs,
            "loss_val" : str(metrics_val[0]),
            "accuracy_val" : str(metrics_val[1]),
            "uar_val" : str(metrics_val[2]),
            "loss_test" : str(metrics_test[0]),
            "accuracy_test" : str(metrics_test[1]),
            "uar_test" : str(metrics_test[2]),
        }

        self.model.save('../model/'+str(id_model))
        with open('../model_registry.json', 'w') as outfile:
            json.dump(data, outfile)





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


class CNNModelLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CNNModelLayer,self).__init__()

        self.cnn1 = Conv2D(filters=4,kernel_size=3,activation="relu")
        self.do1 = Dropout(rate=0.2)
        

        self.cnn2 = Conv2D(filters=16,kernel_size=3,activation="relu")
        self.do2 = Dropout(rate=0.5)
        self.bn1 = BatchNormalization()
        

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = Dense(units=32,activation="relu")
        self.do3 = Dropout(rate=0.5)

        self.dense2 = Dense(units=16,activation="relu")
        self.do4 = Dropout(rate=0.5)
        self.bn2 = BatchNormalization()

        self.dense3 = Dense(units=2,activation="softmax")

    def __call__(self, inputs):

        inputs = tf.expand_dims(inputs, axis=-1)

        
        x = self.do1(self.cnn1(inputs))
        x = self.do2(self.cnn2(x))
        x = self.bn1(x)

        x = self.flat(x)

        x = self.do3(self.dense1(x))
        x = self.do4(self.dense2(x))
        x = self.bn2(x)

        x = self.dense3(x)


        return x


class AttentionModelLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.lstm = LSTM(60, return_sequences = True)
        self.ave_dropout1 = Dropout(rate=0.2)
        self.bn1 = BatchNormalization()
        
        self.ave_dense1 = Dense(50)
        self.ave_dropout2 = Dropout(rate=0.5)
        self.bn2 = BatchNormalization()
        
        
        self.ave_dense2 = Dense(1)
        self.ave_dropout3 = Dropout(rate=0.5)
        
        self.ave_softmax = Softmax()
        self.ave_permute = Permute([2,1])
        self.output_lambda = Lambda(lambda layer: K.sum(layer, axis = -1))
        self.multiply = Multiply()
        self.layer_output = Dense(CLASS_NUMBER)
        
    def __call__(self, inputs):
        
        x = self.lstm(inputs)
        #ave = Attention Vector Estimation
        ave = x
        #ave = self.ave_dropout1(x)
        #ave = self.bn1(ave)
        
        ave = self.ave_dense1(ave)
        #ave = self.ave_dropout2(ave)
        ave = self.bn2(ave)
        
        
        ave = self.ave_dense2(ave)
        #ave = self.ave_dropout3(ave)
        
        ave = self.ave_softmax(ave)
        ave = self.ave_permute(ave)
        
        output_attention =  Lambda(lambda layer: K.sum(layer, axis = -1))(x)
        output_attention = self.multiply([output_attention, ave])
        
        return self.layer_output(output_attention)


class TD_melfilter(tf.keras.layers.Layer):
    def __init__(self):
        self.cnn1 = Conv2D(filters=4,kernel_size=3,activation="relu")

    def __call__(self, inputs):
        print("rouge")
        x = tf.keras.layers.Conv1D(filters = 80, kernel_size = 400, activation='relu',
                                bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_TDmel)(inputs)

        #For the 64 filters:
        # x = tf.keras.layers.Conv1D(filters = 128, kernel_size = 400, activation='relu',
        #                         bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_TDmel)(inputs)
        #compute L2 Norm 
        print("x",x.shape)
        
        a = x[:,:,::2]#even elements (real part)
        b = x[:,:,1::2]#uneven elements (imaginary part)
        y = K.sqrt(a+b)#norm of elements (only 40 channels now)

        #apply haning window separately on each 40 channels (need to repmat hanning and do grouped conv)
        print("y",y.shape)
        y = tf.keras.layers.Conv1D(filters = 40, kernel_size = 400, groups = 40, strides = 160, activation='relu',
                               bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_Hanning)(y)

        #For the 64 filters :
        # y = tf.keras.layers.Conv1D(filters = 64, kernel_size = 400, groups = 64, strides = 160, activation='relu',
        #                        bias_initializer = tf.keras.initializers.Zeros(), kernel_initializer = init_Hanning)(y)
        y = K.abs(y)
        y = K.log(1+y)
        print("bleu")


        return y
    


if __name__ == '__main__':

    # Command example :
    # python model.py -frontEnd melfilt -backEnd cnn -normalization pcen -batch_size 8 -epochs 20

    
    
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
	

    parser.add_argument('-frontEnd',choices=["LLD","melfilt","TDfilt"],nargs='?',type=str,default="TDfilt")
    parser.add_argument('-backEnd',choices=["attention",'cnn'],nargs='?',type=str,default="attention")
    parser.add_argument('-normalization',choices=["log","mvn","pcen","learn_pcen","none"],nargs='?',type=str,default="log")
    parser.add_argument('-loss',choices=["normal_ce","weighted_ce"],nargs='?',type=str,default="normal_ce")
    parser.add_argument('-lr',nargs='?',type=float,default=0.001)
    parser.add_argument('-batch_size',nargs='?',type=int,default=1)
    parser.add_argument('-epochs',nargs='?',type=int,default=10)
    parser.add_argument('-n_ex', nargs='?', type=int, default=1)
    
    args = parser.parse_args()

    DATA_PATH = "../data/"
    set_ = args.normalization
    path = DATA_PATH + set_
    
    model1 = FullModel(args)
    _ = model1.build(tf.keras.optimizers.Adam(learning_rate=args.lr))
    
    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()))
    
    start = time()
    history = model1.train()
    print("Training duration = {:.2f} minutes".format((time()-start)//60))
    
    X_test = np.load(path+"_test.npy",allow_pickle=True)

    y_test = np.load(DATA_PATH+"y_test.npy",allow_pickle=True)
    y_test = tf.keras.utils.to_categorical(y_test.astype(int),num_classes=2)
    
    X_val = np.load(path+"_val.npy",allow_pickle=True)

    y_val = np.load(DATA_PATH+"y_val.npy",allow_pickle=True)
    y_val = tf.keras.utils.to_categorical(y_val.astype(int),num_classes=2)


    metrics_test = model1.model.evaluate(X_test,y_test)
    metrics_val = model1.model.evaluate(X_val,y_val)#model1.model.evaluate(X_val,y_val)
    
    #model1.save(np.round(metrics_val,4),np.round(metrics_test,4))


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim((0,args.epochs))
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()