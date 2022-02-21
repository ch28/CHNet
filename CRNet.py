import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda, multiply, concatenate
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

# Bulid CRNet
def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y

def encoder(x):    
    # encoder
    sidelink = x

    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Conv2D(2, (1, 9), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    x = Conv2D(2, (9, 1), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    sidelink = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(sidelink)
    sidelink = add_common_layers(sidelink)

    x = concatenate([x, sidelink], axis = 1)

    x = Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear')(x) # Note that CRNet has no quantization operation, so the linear activation function is used

    return encoded
    
def decoder(encoded):
    # decoder
    x = Dense(img_total, activation='linear')(encoded) # Due to the settings in CRNet, the linear activation function is adopted
    x = Reshape((img_channels, img_height, img_width,))(x)

    x = Conv2D(2, (5, 5), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    
    # CRBlock
    for i in range(2):
        sidelink = x
        shortcut = x
        
        x = Conv2D(7, (3, 3), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        x = Conv2D(7, (1, 9), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        x = Conv2D(7, (9, 1), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)

        sidelink = Conv2D(7, (1, 5), padding='same', data_format="channels_first")(sidelink)
        sidelink = add_common_layers(sidelink)
        sidelink = Conv2D(7, (5, 1), padding='same', data_format="channels_first")(sidelink)
        sidelink = add_common_layers(sidelink)

        x = concatenate([x, sidelink], axis = 1)

        x = Conv2D(2, (1, 1), padding='same', data_format="channels_first")(x)
        x = add_common_layers(x)
        
        x = add([x, shortcut])

    x = Activation('sigmoid')(x)

    return x

image_tensor = Input(shape=(img_channels, img_height, img_width))
codewords_vector = Input(shape=(encoded_dim,))

encoder = Model(inputs=[image_tensor], outputs=[encoder(image_tensor)])
decoder = Model(inputs=[codewords_vector], outputs=[decoder(codewords_vector)])
autoencoder = Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])
autoencoder.compile(optimizer='adam', loss='mse')
print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())       

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

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        
file = 'CRNet_'+(envir)+'_dim'+str(encoded_dim)

path = 'result/TensorBoard_%s' %file

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

history = LossHistory()

callbacks = [history, TensorBoard(log_dir = path)]

autoencoder.fit_generator(generator=generator(batch_size,x_train), 
                          steps_per_epoch=int(x_train.shape[0]/batch_size), 
                          epochs=epochs, 
                          validation_data=generator_val(batch_size,x_val),
                          validation_steps=int(x_val.shape[0]/batch_size),
                          callbacks=callbacks
                          )

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

print("x_test_C shape is ", x_test_C.shape)
power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("MSE is ", 10*math.log10(np.mean(mse)))
print("NMSE is ", 10*math.log10(np.mean(mse/power)))

