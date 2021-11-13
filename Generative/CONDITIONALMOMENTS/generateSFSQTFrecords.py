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
from tensorflow.keras.models import load_model
from myProgressBar import printProgressBar

par.printRoot('GENERATE TF RECORD WITH SUBFILTER SQUARED')

# Filenames to read
filenameTrain = 'data/Mom1kseTrain.tfrecord'
filenameTest = 'data/Mom1kseTest.tfrecord'
model = load_model('weight1KSE/WeightsSC_filt_4_blocks_2/best.h5')

# Initialize the tf dataset to read
dsTrain = tf.data.TFRecordDataset(filenameTrain)
dsTrain = dsTrain.map(tfu._mom_parse_function)  # parse the record
dsTest = tf.data.TFRecordDataset(filenameTest)
dsTest = dsTest.map(tfu._mom_parse_function)  # parse the record

# Filename to write
dataPath = filenameTrain.split('/')
dataPath[-1] = 'Mom2' + dataPath[-1]
filenameToWriteTrain = os.path.join(*dataPath)

dataPath = filenameTest.split('/')
dataPath[-1] = 'Mom2' + dataPath[-1]
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
    for qoi, data in dsTrain:
        # ~~~~ Prepare the data
        n_qoi  = qoi.shape[1]
        n_data  = data.shape[1]

        # Create the subfilter field
        A = model.predict(np.reshape(qoi,(1,n_qoi,1)))
        subfiltFieldSq = (data - A)**2 

        # ~~~~ Write the data
        tf_example = tfu.mom2_example(counter,n_data,n_qoi,bytes(qoi),bytes(subfiltFieldSq))
        writer.write(tf_example.SerializeToString())

        counter += 1
        printProgressBar(counter, nSnapTrain, prefix = 'Output snapshot Train ' + str(counter) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50)

printProgressBar(0, nSnapTest, prefix = 'Output snapshot Test ' + str(0) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50)
with tf.io.TFRecordWriter(filenameToWriteTest) as writer:
    counter=0
    for qoi, data in dsTest:
        # ~~~~ Prepare the data
        n_qoi  = qoi.shape[1]
        n_data  = data.shape[1]

        # Create the subfilter field
        A = model.predict(np.reshape(qoi,(1,n_qoi,1)))
        subfiltFieldSq = (data - A)**2 
   
        # ~~~~ Write the data
        tf_example = tfu.mom2_example(counter,n_data,n_qoi,bytes(qoi),bytes(subfiltFieldSq))
        writer.write(tf_example.SerializeToString())

        counter += 1
        printProgressBar(counter, nSnapTest, prefix = 'Output snapshot Test ' + str(counter) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50)

