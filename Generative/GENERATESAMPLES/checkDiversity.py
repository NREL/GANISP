import sys
sys.path.append('util')
from plotsUtil import *
import numpy as np
import os

def qoiFn(x):
    mean = np.mean(x)
    return np.sum((x-mean)**2)/len(x)
def qoiFnAll(x):
    mean = np.mean(x,axis=2,keepdims=True)
    return np.sum((x-mean)**2,axis=2, keepdims=True)/x.shape[2]

dataOutRootFolder = 'data_out'
models = [15] 
figFolder = 'Figures'
os.makedirs(figFolder,exist_ok=True)


for model in models:

    dataFolder = dataOutRootFolder + '/model' + str(model)
 
    gen_data = np.load(dataFolder+'/test_gen.npy')
    true_data = np.load(dataFolder+'/test_data.npy')
    qoi_data = np.load(dataFolder+'/test_qoi.npy')
    qoi_gen = qoiFnAll(gen_data)

    ndat = qoi_gen.shape[0]
    nrep = qoi_gen.shape[1]
    ndim = true_data.shape[1]

    print('model'+str(model))


    fig = plt.figure()
    target = np.reshape(qoi_data,(ndat,))
    mean = np.squeeze(np.mean(gen_data,axis=1))
    std = np.squeeze(np.std(gen_data,axis=1))
    plt.plot(target,np.mean(std,axis=1),'o',color='k',label='Generated')
    A = np.load('data/aprioriMom.npz')
    plt.plot(A['qoi'],A['std'],'x',color='k',label='A priori')
    prettyLabels('Q', r'$\langle \sqrt{\mathbb{E}(\xi_i^2|Q)} \rangle_x$', 30)
    plotLegend(25)
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.savefig(figFolder+'/divLoss.png')
    plt.savefig(figFolder+'/divLoss.eps')


plt.show()

    
