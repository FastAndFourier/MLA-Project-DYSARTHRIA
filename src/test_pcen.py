import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Softmax, Input, Permute, Multiply, Lambda, Dropout, Conv2D, BatchNormalization
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from time_mfb import init_Hanning, init_TDmel

from blocks import *

x_train = np.load("../data/mfsc_train.npy")[0:1]
y_train = np.load("../data/y_train.npy")[0:1]

# #Works with (2,10,5) / (2,25,50)
# # x_train = np.random.random((2,64,251))
# # y_train = np.load("../data/y_train.npy")[0:2]

# # print(x_train[0].shape)

inputs = tf.keras.Input(shape=(x_train[0].shape))
outputs = CustomLayerPCEN2()(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer="SGD", loss="mse")
model.fit(x=tf.convert_to_tensor(x_train), y=tf.convert_to_tensor(y_train), epochs =1, batch_size = 1, verbose = 2)


# a= model.get_weights()

# for i in range(len(a)):
#     print(a[i])



# #============Proof that it works==========
# s = 0.5
# eps = 1e-6
# delta = 2.
# alpha = 0.98
# r=0.5
# M = []

# print(x_train[0])
# #Loop on each feature (corresponding to each frequency in the mfcc) and each timestep
# for i in range(64):
#     prec_mean, current_mean = 0., 0.
#     M_row = []
#     for j in range(251):
#     #First timestep, there is no previous value, so we set prec_mean to zero
#         if j == 0:
#             prec_mean = 0
#         else:
#             prec_mean = current_mean
#         #Compute the current mean based on the previous mean and the current value
#         current_mean = (1-s)*prec_mean + s*x_train[0][i,j]
#         M_row.append(current_mean)
#     #Compute the value of the PCEN for each t, f
#     M.append(M_row)

# Madde = tf.math.add(M, eps)
# print("Madde", Madde)

# Mepow = tf.sign(Madde)*tf.pow(tf.abs(Madde), alpha)
# print("Mepow", Mepow)
# EonM = tf.math.divide(x_train[0], Mepow)
# print("EonM=", EonM)
# EonM_delta = tf.math.add(EonM, delta)
# print("EonM_delta", EonM_delta)
# deltapow = tf.math.pow(delta, r)
# print("deltapow", deltapow)
# term = tf.math.pow(EonM_delta, r)
# print("term",term)
# pcen = tf.math.subtract(term, np.double(deltapow))

# print(x_train[0])
# print(pcen)

# import numpy as np
# import matplotlib.pyplot as plt

# plt.figure(1)
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.show()

# plt.figure(2)
# plt.imshow(pcen)
# plt.colorbar()
# plt.show()


# pcen_lib=librosa.pcen(S=x_train[0], sr=16000, gain=0.98, bias=2, power=0.5, eps=1e-6, b=0.5)

# plt.figure(3)
# plt.imshow(pcen_lib)
# plt.colorbar()
# plt.show()


