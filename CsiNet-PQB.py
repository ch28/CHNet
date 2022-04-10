#-----------------------------------------------------------------------------------+
# Author: Xin Liang                                                                 |
# Time Stamp: Jan 4, 2021                                                           |
# Affiliation: Beijing University of Posts and Telecommunications                   |
# Email: liangxin@bupt.edu.cn                                                       |
#-----------------------------------------------------------------------------------+
#                             *** Open Source Code ***                              |
#-----------------------------------------------------------------------------------+
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda, multiply, Layer
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import scipy.io as sio 
import numpy as np
import math
import time
import hdf5storage # load Matlab data bigger than 2GB
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

envir = 'indoor' # 'indoor' or 'outdoor'

# training params
epochs = 2000
batch_size = 200

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  # valid options are 32, 64, 128, 256, 512
bits = 4 # valid options are 2, 3, 4, 5
B = bits

alpha = 1
k = 1

a = 2 # a >= 2, which represents the proximity to impulse function
C = 0.4439938161680786 # Numerical intergration of the proposed approximate gradient
lbd = 1 # coefficient of custom gradient

tf_pi = tf.constant(math.pi)

def my_sigmoid(x):
    x = 1/(1+tf.exp(-alpha*x))
    return x

def sigmoid_inverse(x):
    x = (tf.log(x) - tf.log(1-x))/alpha # x range from 0 to 1
    return x

@tf.custom_gradient
def sigmoid_inverse_op(x):
    result = tf.cast(sigmoid_inverse(x), dtype=tf.float32)
    
    def custom_grad(dy):
        grad = dy * 0.10*(1/x + 1/(1-x))/alpha
        return grad
    return result, custom_grad
    
class sigmoid_inverse_layer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(sigmoid_inverse_layer, self).__init__()
    def call(self, x):
        return sigmoid_inverse_op(x)
    def get_config(self):
        base_config = super(sigmoid_inverse_layer, self).get_config()
        return base_config


# Quantization and Dequantization Layers Defining
@tf.custom_gradient
def QuantizationOp(x):

    xx = tf.clip_by_value(x, 0.5/tf.cast((2**B),tf.float32), 1-(0.5/tf.cast((2**B),tf.float32)))    
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(xx * step - 0.5)), dtype=tf.float32)

    def custom_grad(dy):
        my_step = tf.cast((2**B),tf.float32)
        x_expand = tf.cast(my_step*x, tf.float32)
        y = tf.cast(x_expand-0.5 - tf.round(x_expand-0.5), dtype=tf.float32) # -0.5, 0.5

        y = tf.clip_by_value(tf.abs(y), 0, 1/a-1e-25)

        grad = dy * lbd*(1/C)*1.0*a*(tf.exp(-1/(1-tf.square(a*y))) - tf.exp(-1/tf.square(a*1e-25)))
        return grad 
    return result, custom_grad

class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(QuantizationLayer, self).__init__()
    def call(self, x):
        x = tf.clip_by_value(x, 0., 1.) # ensure sigmoid function
        quantized = QuantizationOp(x)
        return quantized
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        return base_config

def DequantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)
    return result

class DequantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DequantizationLayer, self).__init__()
    def call(self, x):
        dequantized = DequantizationOp(x, self.B)
        return dequantized
    def get_config(self):
        base_config = super(DequantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config

def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    return y

def residual_block_decoded(y):
    shortcut = y
    y = layers.Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)
    
    y = layers.Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)
    
    y = layers.Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = layers.BatchNormalization()(y)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

def encoder_network(x):    
    # encoder
    x = layers.Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    
    x = layers.Reshape((img_total,))(x)
    encoded = layers.Dense(encoded_dim, activation='linear')(x) # encoded result 
    
    # quantization
    encoded = layers.Lambda(my_sigmoid)(encoded)
    encoded = QuantizationLayer()(encoded)
    return encoded
    
def decoder_network(encoded):
    # de-quantization
    encoded = DequantizationLayer(bits)(encoded)
    encoded = sigmoid_inverse_layer()(encoded)
    encoded = layers.Reshape((encoded_dim,))(encoded)
    # decoder
    x = layers.Dense(img_total, activation='linear')(encoded)
    x = layers.Reshape((img_channels, img_height, img_width,))(x)
        
    for i in range(residual_num):
        x = residual_block_decoded(x)
    
    x = layers.Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x

encoder_input = keras.Input(shape=(img_channels, img_height, img_width))
decoder_input = keras.Input(shape=(encoded_dim,))

autoencoder_input = keras.Input(shape=(img_channels, img_height, img_width))

encoder = keras.Model(inputs=[encoder_input], outputs=[encoder_network(encoder_input)])
decoder = keras.Model(inputs=[decoder_input], outputs=[decoder_network(decoder_input)])
autoencoder = keras.Model(inputs=[autoencoder_input], outputs=[decoder(encoder(autoencoder_input))])
autoencoder.compile(optimizer='adam', loss='mse')

print(encoder.summary())
print(decoder.summary())


# Data loading
dataset_path = '../data'

if envir == 'indoor':
    data_path = dataset_path + '/DATA_Htrainin.mat'
    mat = hdf5storage.loadmat(data_path)
    x_train = mat['HT'] # array

    data_path = dataset_path + '/DATA_Hvalin.mat'
    mat = hdf5storage.loadmat(data_path)
    x_val = mat['HT'] # array

    data_path = dataset_path + '/DATA_Htestin.mat'
    mat = hdf5storage.loadmat(data_path)
    x_test = mat['HT'] # array

elif envir == 'outdoor':
    data_path = dataset_path + '/DATA_Htrainout.mat'
    mat = hdf5storage.loadmat(data_path)
    x_train = mat['HT'] # array

    data_path = dataset_path + '/DATA_Hvalout.mat'
    mat = hdf5storage.loadmat(data_path)
    x_val = mat['HT'] # array

    data_path = dataset_path + '/DATA_Htestout.mat'
    mat = hdf5storage.loadmat(data_path)
    x_test = mat['HT'] # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        
file = 'CsiNet_PQB'+(envir)+'_dim'+str(encoded_dim)+'_'+str(bits)+'bits'

path = 'result/TensorBoard_%s' %file

save_dir = os.path.join(os.getcwd(), '/result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

history = LossHistory()


callbacks = [history, tf.keras.callbacks.TensorBoard(log_dir = path)]

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)
                

outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)

# Testing data

tStart = time.time()
x_hat = autoencoder.predict([x_test])
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("With", bits, "quantization bits")
print("MSE is ", 10*math.log10(np.mean(mse)))
print("NMSE is ", 10*math.log10(np.mean(mse/power)))



