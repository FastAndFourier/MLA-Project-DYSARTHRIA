import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import build_model, uar_metric


def restore_model_batch(path,frontEnd):

    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-frontEnd',choices=["TDfilt","melfilt","LLD"],nargs='?',type=str,default=frontEnd)
    parser.add_argument('-normalization',choices=["log","mvn","pcen","learn_pcen","none"],nargs='?',type=str,default="log")
    parser.add_argument('-lr',nargs='?',type=float,default=0.0001)
    parser.add_argument('-batch_size',nargs='?',type=int,default=1)
    parser.add_argument('-epochs',nargs='?',type=int,default=5)
    parser.add_argument('-decay',nargs='?',type=bool,default=False)
    
    args = parser.parse_args()

    model = build_model(args,tf.keras.optimizers.SGD())
    model.load_weights(path)  

    # model_old = tf.keras.models.load_model(path,compile=False)
    # model = build_model(args,tf.keras.optimizers.SGD())
    # w = model_old.get_weights()
    # model.set_weights(w)

    return model

def load_model(path,type_,evaluate=True,normalization=None):

    # if type_ in ["melfilt","LLD"]:
    model = restore_model_batch(path,type_)
    # else:
    #     model = tf.keras.models.load_model(path,compile=False)

    model.compile(loss = "binary_crossentropy", metrics = ['binary_accuracy',uar_metric])

    if type_ == "melfilt":
        X_test = np.load("../data/"+normalization+"_test.npy").transpose([0,2,1])
        X_val = np.load("../data/"+normalization+"_val.npy").transpose([0,2,1])
        batch = 1
    elif type_ == "TDfilt":
        X_test = np.load("../data/x_test.npy")
        X_val = np.load("../data/x_val.npy")
        batch=model.input.shape[0]
    else:
        X_test = np.load("../data/full_lld_is09_test.npy")
        X_val = np.load("../data/full_lld_is09_val.npy")
        batch = 1

    y_test = np.load("../data/y_test.npy")
    y_val = np.load("../data/y_val.npy")


    if evaluate:
        y_test = tf.keras.utils.to_categorical(y_test.astype(int),num_classes=2)
        y_val = tf.keras.utils.to_categorical(y_val.astype(int),num_classes=2)

        metrics_test = model.evaluate(X_test,y_test,batch_size=batch)
        metrics_val = model.evaluate(X_val,y_val,batch_size=batch)

        print(metrics_test,metrics_val)

    return model


def visualize_attention(x,pcen,model,type_):

    if type_ == "TDfilt":
        func_att = K.function(model.input,model.layers[20].output)
    elif type_ == "melfilt":
        func_att = K.function(model.input,model.layers[10].output)
        x = x.transpose([1,0])
    else:
        func_att = K.function(model.input,model.layers[10].output)

    y_pred = K.argmax(model.predict(x[np.newaxis]))

    print("Predicted class : ",y_pred[0].numpy())
    att_vec = func_att(x[np.newaxis])

    plt.figure(figsize=(15,8))
    plt.imshow(pcen)
    plt.plot(-(att_vec[0]*300)+32,color='red')
    plt.show()


if __name__ == "__main__":

    frontEnd = "LLD" # TDfilt, melfilt or LLD
    norm = "pcen"
    
    pcen = np.load("../data/pcen_test.npy")
    x = np.load("../data/x_test.npy")


    # !! If frontEnd == "TDfilt", batch_1=False
    model = load_model("../log/weights_LL.h5",frontEnd,normalization=norm,evaluate=True)

    #model.save_weights("../log/weights_LL.h5")

    # index = np.random.randint(low=0,high=len(x))

    # visualize_attention(x[index],pcen[index],model,frontEnd)
