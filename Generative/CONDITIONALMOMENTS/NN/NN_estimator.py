from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflowUtils as tfu
import sys
import scipy.io as sio 


# FROM : https://github.com/krasserm/super-resolution/blob/master/model/srgan.py
def res_block(x_in,num_filters):
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x_in)
    x = PReLU()(x)
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x_in, x])
    return x

# FROM : https://github.com/krasserm/super-resolution/blob/master/model/srgan.py
def upsample(x_in, num_filters):
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x_in)
    return PReLU()(x)

def make_SRNN_SC_model(Case):
    if Case['n_data']%Case['n_upsample'] > 0:
        print('n_data should be divisible by n_upsample')
        print('n_data =',Case['n_data'])
        print('n_upsample =',Case['n_upsample'])
        sys.exit()

    num_filters = Case['numFilters']
    numBlocks = Case['numBlocks']

    x_in = Input(shape=(Case['n_qoi'],1))
    x = Flatten()(x_in)
    x = Dense(Case['n_upsample'])(x)
    x = PReLU()(x)
    x = Reshape((Case['n_upsample'],1))(x)
    x = Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = x_1 = PReLU()(x)

    for _ in range(numBlocks):
        x = res_block(x, num_filters)

    x = Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x_1, x])
    x = upsample(x,Case['n_data']//Case['n_upsample'])

    x = Reshape((Case['n_data'], 1))(x)

    return Model(x_in, x)

def makeModel(Case):
    SRNN = make_SRNN_SC_model(Case)
    SRNN.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])
    SRNN.summary()
    return SRNN


def train(SRNN,Case):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    EPOCHS = 10
    BATCH_SIZE=min(512 ,Case['nSnapTrain'])   
    Case['dsTrain'] = Case['dsTrain'].shuffle(100).repeat(EPOCHS).batch(BATCH_SIZE)
    Case['dsTest'] = Case['dsTest'].batch(BATCH_SIZE)
    WeightFolder = Case['weightFolder']+'/WeightsSC_filt_'+str(Case['numFilters'])+'_blocks_'+str(Case['numBlocks'])
    path = WeightFolder.split('/')
    for i in range(len(path)):
         directory = os.path.join(*path[:i+1])
         os.makedirs(directory,exist_ok=True)

    #mc = tf.keras.callbacks.ModelCheckpoint(WeightFolder+'/weights{epoch:08d}.h5', period=100)
    mc = tf.keras.callbacks.ModelCheckpoint(WeightFolder+'/best.h5', save_best_only=True, mode='min')
    csv_logger = tf.keras.callbacks.CSVLogger(WeightFolder +'/training.log')
    

    history = SRNN.fit(
        Case['dsTrain'],
        steps_per_epoch=Case['nSnapTrain'] // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=Case['dsTest'],
        validation_steps=Case['nSnapTest'] // BATCH_SIZE,
        callbacks = [mc,csv_logger]
    )
    


