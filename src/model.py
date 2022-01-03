import numpy as np
import tensorflow as tf
import os, json

from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Input, Dropout, BatchNormalization, Multiply
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import argparse

from blocks import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from time import time

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()





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



def build_model(args,optimizer):

    if args.frontEnd == "melfilt":
        input_ = Input(shape=[251,64],name="input") #,batch_size=args.batch_size
        x = input_
    elif args.frontEnd == "TDfilt":
        input_ = Input(shape=[40000,1],name="input") #,batch_size=args.batch_size
        x = TD_filt(input_)
    elif args.frontEnd == "LLD":
        input_ = Input(shape=[251,32],name="input") #,batch_size=args.batch_size
        x = input_
    else:
        print("Unknown frontEnd")
        input_ = None
        return None

    if args.normalization == "learn_pcen":
        x = CustomLayerPCEN2()(x)
        # x = Dropout(0.2)(x)

    x = LSTM(60, return_sequences = True)(x)

    att = Attention(x)

    x = Lambda(lambda layer: K.sum(layer, axis = -1))(x)
    x = Multiply()([x,att])
    
    output = Dense(2, activation='sigmoid')(x)

    decay_step = int(10)*(6000/args.batch_size)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.lr,                                               
                                                                 decay_steps=decay_step,
                                                                 decay_rate=0.80,
                                                                 staircase=True)
    if args.decay:      
        optimizer.learning_rate = lr_schedule
    else:
        optimizer.learning_rate = args.lr
    optimizer.momentum = 0.98
    
    model = Model(input_, output)
    model.compile(optimizer=optimizer, loss = "binary_crossentropy", metrics = ['binary_accuracy', uar_metric], run_eagerly=True)


    return model


def train_model(args,model,path,log_path):

    if args.frontEnd == 'melfilt':

        if args.normalization in ["none","learn_pcen"]:
            X_train = np.load(path+'mfsc_train.npy',allow_pickle=True).transpose([0,2,1])
            X_val = np.load(path+'mfsc_val.npy',allow_pickle=True).transpose([0,2,1])      
        else:
            X_train = np.load(path+args.normalization+'_train.npy',allow_pickle=True).transpose([0,2,1])
            X_val = np.load(path+args.normalization+'_val.npy',allow_pickle=True).transpose([0,2,1])
            
    elif args.frontEnd == 'TDfilt':
        X_train = np.load(path+'x_train.npy',allow_pickle=True)
        X_val = np.load(path+'x_val.npy',allow_pickle=True)

    elif args.frontEnd == 'LLD':
        X_train = np.load(path+'full_lld_is09_train.npy',allow_pickle=True)
        X_val = np.load(path+'full_lld_is09_val.npy',allow_pickle=True)




    y_train = np.load(path+"y_train.npy")
    y_val = np.load(path+"y_val.npy")
        
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=[0,1],y=y_train)
    
    class_weights = dict(zip([0,1], class_weights))

    y_train = tf.keras.utils.to_categorical(y_train.astype(int),num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val.astype(int),num_classes=2)

    #earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_uar_metric', patience=20, mode="max", restore_best_weights=True)
    
    
    id_ = np.random.randint(low=0, high=10**8)
    os.mkdir(log_path+str(id_))
    checkPoint = tf.keras.callbacks.ModelCheckpoint(filepath=log_path+str(id_)+"/",
                                                    monitor='val_uar_metric',
                                                    save_weights_only=True,
                                                    mode='max',save_best_only=True)
    

    
    model.fit(X_train, y_train, batch_size = args.batch_size, 
              epochs = args.epochs, verbose = 1, validation_data=(X_val,y_val),
              callbacks=[checkPoint],shuffle=True,class_weight=class_weights)


    return model, model.history, X_val, y_val, id_


def save_model(model,id_,metrics_test,metrics_val,path):
    
    os.mkdir(path+str(id_))
    model.save(path+str(id_))

    data = {
        "frontEnd" : args.frontEnd,
        "normalization" : args.normalization,
        "lr" : args.lr,
        "batch_size" : args.batch_size,
        "epochs" : args.epochs,
        "loss_val" : str(metrics_val[0]),
        "accuracy_val" : str(metrics_val[1]),
        "uar_val" : str(metrics_val[2]),
        "loss_test" : str(metrics_test[0]),
        "accuracy_test" : str(metrics_test[1]),
        "uar_test" : str(metrics_test[2]),
    }


    with open(path+str(id_)+'/model_registry.json', 'w') as outfile:
        json.dump(data, outfile)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
	
    DATA_PATH = "../data/"

    parser.add_argument('-frontEnd',choices=["TDfilt","melfilt","LLD"],nargs='?',type=str,default="melfilt")
    parser.add_argument('-normalization',choices=["log","mvn","pcen","learn_pcen","none"],nargs='?',type=str,default="log")
    parser.add_argument('-lr',nargs='?',type=float,default=0.0001)
    parser.add_argument('-batch_size',nargs='?',type=int,default=32)
    parser.add_argument('-epochs',nargs='?',type=int,default=5)
    parser.add_argument('-decay',nargs='?',type=bool,default=False)
    
    args = parser.parse_args()

    path = DATA_PATH + args.normalization
    if args.frontEnd == "LLD":
        path = DATA_PATH + "full_lld_is09"
    if args.frontEnd == "TDfilt":
        path = DATA_PATH + "x"

        
    model = build_model(args,tf.keras.optimizers.SGD(learning_rate=args.lr))
    model.summary()

    start = time()
    model, history, X_val, y_val, id_ = train_model(args,model,DATA_PATH,"../log/")
    print("Duration: {:.2f} minutes".format((time()-start)//60))
    print("ID = ",id_)


    model.load_weights("../log/"+str(id_)+"/")

    X_test = np.load(path+"_test.npy",allow_pickle=True)
    if args.frontEnd == "melfilt":
        X_test = X_test.transpose([0,2,1])
    y_test = np.load(DATA_PATH+"y_test.npy",allow_pickle=True)

   
    y_val_ = K.argmax(y_val,axis=-1)

    y_pred_test = K.argmax(model.predict(X_test))
    y_pred_val = K.argmax(model.predict(X_val))


    print(confusion_matrix(y_val_,y_pred_val))
    print(confusion_matrix(y_test,y_pred_test))

    y_test = tf.keras.utils.to_categorical(y_test.astype(int),num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val.astype(int),num_classes=2)

    metrics_test = model.evaluate(X_test,y_test,batch_size=1)
    metrics_val = model.evaluate(X_val,y_val,batch_size=1)
    
    if args.frontEnd=="TDfilt":
        path_save = '../model_archive_tdfilt/model_'
    elif args.frontEnd=="melfilt":
        path_save = '../model_archive_melfilt/model_'
    else:
        path_save = '../model_archive_LLD/model_'

    tf.keras.models.save_model(model,path_save+str(id_))

        





    