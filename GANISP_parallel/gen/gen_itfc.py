import numpy as np
import tensorflow as tf
from cgan_network import cGAN_NETWORK
import time


def loadGen(modelFileName='G_model/epoch00078/gan'):

    g_n_resBlocks = 16
    g_n_filters = 32
    d_firstNFilt = 8
    N_Z = 16
    model = cGAN_NETWORK(n_qoi=1, n_data=128, n_z=N_Z, rep_size=32,
                                 g_n_resBlocks=g_n_resBlocks, g_n_filters=g_n_filters,d_firstNFilt=d_firstNFilt)
    generator = model.generator
    generator.load_weights(modelFileName)

    return generator

#def diffFn(ugen,uref):
#    uref = np.reshape(uref,(1,ugen.shape[1],1))
#    return np.squeeze(np.std(ugen-uref,axis=1))
#    #return np.squeeze(np.mean(abs(ugen-uref),axis=1))

def diffFn(ugen,uref):
    #print(ugen.shape)
    #print(uref.shape)
    uref = np.reshape(uref,(ugen.shape[0],ugen.shape[1],1))
    #return np.squeeze(np.std(ugen-uref,axis=1))
    #return np.squeeze(np.mean(abs(ugen-uref),axis=1))
    return np.mean(np.linalg.norm(ugen-uref,axis=1)/ugen.shape[1])

def diffFnRank(ugen,uref):
    uref = np.reshape(uref,(1,ugen.shape[1],1))
    return np.linalg.norm(ugen-uref,axis=1)/ugen.shape[1]

def cloneSample(generator,qoiVal,nClone,uref):
    batch_qoi = np.zeros((1,1,1))
    batch_qoi[0,0,0] = (qoiVal - 1.7240014844838842)/0.20744504957106663
    batch_qoi_repeated = np.repeat(batch_qoi, nClone, axis=0)
    batch_z = np.random.uniform(low=-1, high=1, size=[nClone, 16, 1])
    genData = generator([batch_qoi_repeated,batch_z],training=False)*1.3130419828853965 + -5.0281370042055505e-06

    return np.mean(diffFnRank(genData,uref_reshape)), genData 
    

def closeCloneSample(generator,qoiVal,nClone,uref):
    time_s = time.time()
    nrep = 1000
    batch_qoi = np.zeros((1,1,1))
    batch_qoi[0,0,0] = (qoiVal - 1.7240014844838842)/0.20744504957106663
    batch_qoi_repeated = np.repeat(batch_qoi, nrep*nClone, axis=0)
    

    batch_z = np.random.uniform(low=-1.0, high=1.0, size=[nrep*nClone, 16, 1])
    genData = generator([batch_qoi_repeated,batch_z],training=False)*1.3130419828853965 + -5.0281370042055505e-06
 
    diffGen = diffFnRank(genData,uref)
    Ind = np.argpartition(diffGen,nClone-1)
 
    #print('close=',np.amin(diffGen))
    time_e = time.time()
    #print('exec in %.2f s' % (time_e-time_s))
    return np.take(genData,list(Ind[:nClone]),axis=0)

def recursiveCloseCloneSample(generator,qoiVal,nClone,uref):
    time_s = time.time()

    nexplore = 25
    nexploit = max(200,3*nClone)
    nrec = 7

    batch_qoi = np.zeros((1,1,1))
    batch_qoi[0,0,0] = (qoiVal - 1.7240014844838842)/0.20744504957106663
    batch_qoi_repeated = np.repeat(batch_qoi, (nexplore+nexploit), axis=0)
   
    optimz = np.zeros((1,16,1))
    for irec in range(nrec):
        batch_z = np.clip(np.random.uniform(low=-1.0/(1.25**(irec)), high=1.0/(1.25**(irec)), size=[nexploit, 16, 1]) + optimz,-1,1)
        batch_z = np.vstack((batch_z,np.random.uniform(low=-1.0, high=1.0, size=[nexplore, 16, 1])))
        genData = generator([batch_qoi_repeated,batch_z],training=False)*1.3130419828853965 + -5.0281370042055505e-06
        
        diffGen = diffFnRank(genData,uref)
        optimz = batch_z[np.argmin(diffGen)]
        #print(np.amin(diffGen)) 
        #print(optimz)
    Ind = np.argpartition(diffGen,nClone-1,axis=0)[:,0]
     
    #print('rec=' + str(np.amin(diffGen)) + ' nclone=' + str(nClone)) 
    time_e = time.time()
    #print('exec in %.2f s' % (time_e-time_s))
   
    return np.mean(diffGen[Ind[:nClone]]), np.take(genData,list(Ind[:nClone]),axis=0)
