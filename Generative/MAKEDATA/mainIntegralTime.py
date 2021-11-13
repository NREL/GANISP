import sys
sys.path.append('util')
import numpy as np
import myparser as myparser
import simulation as simulation
import data as data
import parallel as par
import time
import monitorTiming as monitorTiming
import postProc as postProc
from plotsUtil import *
from myProgressBar import printProgressBar



def autocorr(x):
    nTime = min(int(x.shape[0]/2),250)
    nDim = x.shape[1]
    nReal = x.shape[2]
    meanx = np.mean(x,axis=(0,2),keepdims=True)
    x = x-meanx
        
    corr = np.zeros(nTime)
    printProgressBar(0, nTime-1, prefix = 'Autocorrelation ' + str(0) + ' / ' +str(nTime-1),suffix = 'Complete', length = 50)
    for itime in range(nTime): 
        xroll = np.roll(x,-itime,axis=0)
        corr[itime] = np.mean(x*xroll) 
        printProgressBar(itime, nTime-1, prefix = 'Autocorrelation ' + str(itime) + ' / ' +str(nTime-1),suffix = 'Complete', length = 50) 
    corr=corr/corr[0]

    for itime in range(nTime):
        if corr[itime]<0.1:
            break     
  
    return corr, itime

# ~~~~ Init
# Parse input
inpt = myparser.parseInputFile()

# Initialize MPI
# Not for now

# Initialize random seed
np.random.seed(seed=42+par.irank)

# Initialize Simulation details
Sim = data.simSetUp(inpt)


# ~~~~ Main
# Run Simulation
Result = simulation.simRun(Sim)


# ~~~~ Monitor
# Timing
monitorTiming.printTiming(Result)


# ~~~~ Post process
# Plot, build CDF
postProc.postProc(Result,Sim)


par.finalize()


# SAVE DATA
if Sim['Simulation name'] == 'KS':
    indexLim = np.argwhere(Result['tt']>50)[0][0]
    autcorrelationCoeff, integralTime = autocorr(Result['uu'][indexLim:,:,:200])
    fig = plt.figure()
    time = np.linspace(0,len(autcorrelationCoeff)*Sim['Timestep'],len(autcorrelationCoeff))
    plt.plot(time,autcorrelationCoeff,color='k',linewidth=3)
    prettyLabels('time',r'$\rho$',14,title=r'Integral time = %.4f s' % (integralTime*Sim['Timestep']))
    plt.show()



