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
    dataFolder = dataOutRootFolder + '/' + '1D-20211113-090211'
 
    gen_data = np.load(dataFolder+'/test_gen.npy')
    true_data = np.load(dataFolder+'/test_data.npy')
    qoi_data = np.load(dataFolder+'/test_qoi.npy')
    qoi_gen = qoiFnAll(gen_data)

    ndat = qoi_gen.shape[0]
    nrep = qoi_gen.shape[1]
    ndim = true_data.shape[1]

    contentLoss = np.mean((qoi_gen-np.reshape(qoi_data,(ndat,1,1,1)))**2)
    print('model'+str(model))
    print('\t contentLoss = ' + str(contentLoss))


    fig = plt.figure()
    target = np.reshape(np.squeeze(np.repeat(qoi_data,nrep,axis=1)),(ndat*nrep,))
    generated = np.reshape(np.squeeze(qoi_gen),(ndat*nrep,))
    targetPlot = np.linspace(np.amin(target)*0.8,np.amax(target)*1.2,100)
    plt.plot(targetPlot,targetPlot,'--',color='k',label='Ideal')
    plt.plot(target,generated,'o',color='k',label='Generated')
    prettyLabels(r'Q$_{input}$', r'Q$_{gen}$', 30)
    plotLegend(25)
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.savefig(figFolder+'/contentLoss.png')
    plt.savefig(figFolder+'/contentLoss.eps')
    plt.show()

### Make movie dir
#movieDir = 'MovieTmp'
#os.makedirs(movieDir,exist_ok=True)
#
## Select a snapshot
#ind = np.random.randint(0,gen_data.shape[0])
#print(ind)
#print ('Theoretical qoi = ', qoi_data[ind,0,0])
#print ('computed qoi = ', qoiFn(gen_data[ind,0,:,0]))
#for i in range(gen_data.shape[1]):
#    x = np.linspace(0,32*np.pi,gen_data.shape[2])
#    plotNPlots(field=[gen_data[ind,i,:,0],true_data[ind,:,0]], x=[x,x], title=[r'$\xi_{gen}$, qoi=%.2f' % qoi_gen[ind,i,0] ,r'$\xi_{true}$, qoi = %.2f' %  qoi_data[ind,0,0]])
#    plt.savefig(movieDir+'/im_'+str(i)+'.png')
#    plt.close()   
#
#makeMovie(gen_data.shape[1],movieDir,'gen.gif',fps=4)


    
