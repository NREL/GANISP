import itertools
import time
import h5py
import sys
import os
import scipy.special
import numpy as np
sys.path.append('partools')
sys.path.append('scitools')
sys.path.append('util')
import parallel as par
from filters import boxFilter2D, upSample2D
import tensorflow as tf
import tensorflowUtils as tfu
from myProgressBar import printProgressBar

par.printRoot('GENERATE TF RECORD WITH SUBFILTER')

# Filenames to read
filenameTrain = 'data/kseTrain.tfrecord'
filenameTest = 'data/kseTest.tfrecord'

# Initialize the tf dataset to read
dsTrain = tf.data.TFRecordDataset(filenameTrain)
dsTrain = dsTrain.map(tfu._parse_function)  # parse the record
dsTest = tf.data.TFRecordDataset(filenameTest)
dsTest = dsTest.map(tfu._parse_function)  # parse the record

# Filename to write
dataPath = filenameTrain.split('/')
dataPath[-1] = 'Mom1' + dataPath[-1]
filenameToWriteTrain = os.path.join(*dataPath)

dataPath = filenameTest.split('/')
dataPath[-1] = 'Mom1' + dataPath[-1]
filenameToWriteTest = os.path.join(*dataPath)


nSnapTrain = 0
for _,_ in dsTrain:
    nSnapTrain += 1
nSnapTest = 0
for _,_ in dsTest:
    nSnapTest += 1


printProgressBar(0, nSnapTrain, prefix = 'Output snapshot Train ' + str(0) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50)
with tf.io.TFRecordWriter(filenameToWriteTrain) as writer:
    counter=0
    for data, qoi in dsTrain:
 
        # ~~~~ Prepare the data
        qoi_snapshot = np.squeeze(qoi.numpy())
        data_snapshot = np.squeeze(data.numpy())
        n_qoi  = qoi.shape[1]
        n_data  = data.shape[1]    

        # ~~~~ Write the data
        tf_example = tfu.mom1_example(counter,n_data,n_qoi,bytes(qoi),bytes(data))
        writer.write(tf_example.SerializeToString())

        counter += 1
        printProgressBar(counter, nSnapTrain, prefix = 'Output snapshot Train ' + str(counter) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50)



printProgressBar(0, nSnapTest, prefix = 'Output snapshot Test ' + str(0) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50)
with tf.io.TFRecordWriter(filenameToWriteTest) as writer:
    counter=0
    for data, qoi in dsTest:
 
        # ~~~~ Prepare the data
        qoi_snapshot = np.squeeze(qoi.numpy())
        data_snapshot = np.squeeze(data.numpy())
        n_qoi  = qoi.shape[1]
        n_data  = data.shape[1]    

        # ~~~~ Write the data
        tf_example = tfu.mom1_example(counter,n_data,n_qoi,bytes(qoi),bytes(data))
        writer.write(tf_example.SerializeToString())

        counter += 1
        printProgressBar(counter, nSnapTest, prefix = 'Output snapshot Test ' + str(counter) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50)
        
