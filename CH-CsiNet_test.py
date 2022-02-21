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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

saved_dim = 512

test_mode = 'multiple_dot' # 'single_dot' or 'multiple_dot'

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
encoded_dim = 512

# Loading Data
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

x_test = x_val

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

file = 'CH-CsiNet_'+(envir)+'_dim'+str(encoded_dim)

path = 'result/TensorBoard_%s' %file

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Testing data
if test_mode == 'single_dot': # Test the NMSE performance of one selected feedback overhead
    start_dim = saved_dim
elif test_mode == 'multiple_dot': # Test the NMSE performance of all valid feedback overhead
    start_dim = 0

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
power = np.sum(abs(x_test_C)**2, axis=1)

for cut_dim in range(start_dim, saved_dim+1):
    def FOCU_enc(x):
        for row in range(0, x.shape[0]):
            x[row, cut_dim:] = 0 # Changeable feedback overhead control unit
        return x

    @tf.custom_gradient
    def FOCU_op(x):
        result = tf.py_func(func=FOCU_enc, inp=[x], Tout=tf.float32)
        def custom_grad(dx):
            grad = tf.sign(tf.abs(result)) # The back-propagation process of the cut part is forbidden
            dy = dx * grad 
            return dy
        return result, custom_grad

    class FOCU(Layer):
        def __init__(self,**kwargs):
            super(FOCU, self).__init__()
        def call(self, x):
            quantized = FOCU_op(x)
            return quantized
        def get_config(self):
            base_config = super(FOCU, self).get_config()
            return base_config

    # Bulid CH-CsiNet
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    def encoder(x):
        # encoder
        x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        
        x = Reshape((img_total,))(x)
        encoded = Dense(encoded_dim, activation='linear')(x) # encoded result

        cut_encoded = FOCU()(encoded)

        return cut_encoded
        
    def decoder(cut_encoded):
        # decoder
        x = Dense(img_total, activation='linear')(cut_encoded)
        x = Reshape((img_channels, img_height, img_width,))(x)
            
        for i in range(residual_num):
            x = residual_block_decoded(x)
        
        x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

        return x


    image_tensor = Input(shape=(img_channels, img_height, img_width))
    codewords_vector = Input(shape=(encoded_dim,))

    encoder = Model(inputs=[image_tensor], outputs=[encoder(image_tensor)])
    decoder = Model(inputs=[codewords_vector], outputs=[decoder(codewords_vector)])
    autoencoder = Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])

    outfile = 'result/%s_model.h5' % file
    autoencoder.load_weights(outfile)

    tStart = time.time()
    x_hat = autoencoder.predict(x_test)
    tEnd = time.time()
    print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

    mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
    nmse = 10*math.log10(np.mean(mse/power))

    print("In "+envir+" environment")
    print("When dimension is", cut_dim)
    print("NMSE is ", nmse)


