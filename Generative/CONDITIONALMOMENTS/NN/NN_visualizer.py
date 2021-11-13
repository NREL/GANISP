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
import tensorflow as tf
import tensorflowUtils as tfu
from plotsUtil import *
from myProgressBar import printProgressBar

def plotMoments(Case):
   
    # Only root processor does something   
    if not par.irank == par.iroot:
        return 

    # Filename
    dataPath = Case['dataFilenameTest'].split('/')
    indexPrefix = dataPath[-1].index('.tfrecord')
    dataPath[-1] = 'modelNN' + '_' + dataPath[-1][:indexPrefix] + "_diversity.tfrecord"
    filenameToRead = os.path.join(*dataPath)

    # Initialize the tf dataset
    dsM = tf.data.TFRecordDataset(filenameToRead)
    dsM = dsM.map(tfu._parse_diversity)  # parse the record

    # Size of dataset
    n_qoi = Case['n_qoi']    
    n_data = Case['n_data']   

     
    nSnap = min(Case['nSnapTest'],1000)   
  
    # ~~~~ Read dataset
    counter = 0
    printProgressBar(0, nSnap, prefix = 'Movie snapshot ' + str(0) + ' / ' +str(nSnap),suffix = 'Complete', length = 50)
    movieDir  = 'MovieTmp'
    os.makedirs(movieDir,exist_ok=True)
    qoiPlot = []
    meanStd = []
    for data, qoi, mean, std in dsM:
         index = counter
         
         # ~~~~ Log advancement
         printProgressBar(counter+1, nSnap, prefix = 'Movie snapshot ' + str(counter+1) + ' / ' +str(nSnap),suffix = 'Complete', length = 50)
         # ~~~~ Prepare the data
         qoi_snapshot = np.squeeze(qoi.numpy())
         data_snapshot = np.squeeze(data.numpy())
         mean = np.squeeze(mean.numpy())
         std = np.squeeze(std.numpy())

         x = np.linspace(0,32*np.pi,len(data_snapshot))
 
         plotNPlots(field=[data_snapshot,mean,std], x=[x,x,x], title=[r'$\xi_{HR}$',r'$E(\xi_{HR}|qoi=%.2f)$'% qoi_snapshot,r'$\sigma(\xi_{HR}|qoi=%.2f)$'%qoi_snapshot]) 
         plt.savefig(movieDir+'/im_'+str(index)+'.png')
         plt.close()
        
         # ~~~~ Collect data for external plot
         qoiPlot.append(qoi_snapshot)
         meanStd.append(np.mean(std))
 
         counter += 1
        

         if counter==nSnap:
             break
    makeMovie(nSnap,movieDir,'conditionalMoments_modelNN.gif',fps=4)

    fig = plt.figure()
    plt.plot(qoiPlot,meanStd,'o',color='k')
    prettyLabels('qoi','Ave std',14)
    plt.savefig('summary.png')
    plt.close()
