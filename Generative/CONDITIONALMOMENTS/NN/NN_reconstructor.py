from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("util")
sys.path.append("scitools")
sys.path.append("partools")
import parallel as par
from filters import boxFilter2D, upSample2D
from plotsUtil import *
import tensorflowUtils as tfu
from tensorflow.keras.models import load_model
from myProgressBar import printProgressBar

def writeTfRecords(Case):
    n_qoi = Case['n_qoi']    
    n_data = Case['n_data']    
    nSnapTrain = Case['nSnapTrain']    
    nSnapTest = Case['nSnapTest']    

    dataPath = Case['dataFilenameTrain'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'modelNN' + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToWrite = os.path.join(*dataPath)
  
    batchedDataSet = Case['dsTrain'].batch(4096) 

    # ~~~~ Write TF RECORD for training data
    if par.irank == par.iroot:
        printProgressBar(0, nSnapTrain, prefix = 'Output snapshot Train ' + str(0) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50) 
        with tf.io.TFRecordWriter(filenameToWrite) as writer:
            counter=0
            for element in batchedDataSet:
       
                qoi = element[0]
                data = element[1]   

                # ~~~~ Prepare the data
                n_batch = qoi.shape[0]
                n_qoi  = qoi.shape[2]
                n_data  = data.shape[2]

                # ~~~~ Prepare the data
                # Create moments
                Mean = np.float64(Case['modelFirstMoment'].predict(np.reshape(qoi,(n_batch,n_qoi,1))))
                Std = np.sqrt(np.clip(Case['modelSecondMoment'].predict(np.reshape(qoi,(n_batch,n_qoi,1))),0,100000))
                

                # ~~~~ Write the data
                for idat in range(n_batch):
                    index = counter
                    tf_example = tfu.diversity_example(counter,bytes(qoi[idat]),n_qoi,bytes(data[idat]),n_data,
                                                       bytes(Mean[idat]),bytes(Std[idat]))
                    writer.write(tf_example.SerializeToString())
    
                    # ~~~~ Log advancement
                    counter=counter+1
                    printProgressBar(counter, nSnapTrain, prefix = 'Output snapshot Train ' + str(counter) + ' / ' +str(nSnapTrain),suffix = 'Complete', length = 50) 




    dataPath = Case['dataFilenameTest'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'modelNN' + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToWrite = os.path.join(*dataPath)
   
    batchedDataSet = Case['dsTest'].batch(4096) 
   
    # ~~~~ Write TF RECORD for testing data
    if par.irank == par.iroot:
        printProgressBar(0, nSnapTest, prefix = 'Output snapshot Test ' + str(0) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50) 
        with tf.io.TFRecordWriter(filenameToWrite) as writer:
            counter=0
            for element in batchedDataSet:
       
                qoi = element[0]
                data = element[1]   

                # ~~~~ Prepare the data
                n_batch = qoi.shape[0]
                n_qoi  = qoi.shape[2]
                n_data  = data.shape[2]

                # ~~~~ Prepare the data
                # Create moments
                Mean = np.float64(Case['modelFirstMoment'].predict(np.reshape(qoi,(n_batch,n_qoi,1))))
                Std = np.sqrt(np.clip(Case['modelSecondMoment'].predict(np.reshape(qoi,(n_batch,n_qoi,1))),0,100000))
                

                # ~~~~ Write the data
                for idat in range(n_batch):
                    index = counter
                    tf_example = tfu.diversity_example(counter,bytes(qoi[idat]),n_qoi,bytes(data[idat]),n_data,
                                               bytes(Mean[idat]),bytes(Std[idat]))
                    writer.write(tf_example.SerializeToString())
    
                    # ~~~~ Log advancement
                    counter=counter+1
                    printProgressBar(counter, nSnapTest, prefix = 'Output snapshot Test ' + str(counter) + ' / ' +str(nSnapTest),suffix = 'Complete', length = 50) 

