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






#plt.imshow(np.transpose(np.transpose(Result['uu'][-2*NBack:-1,:])),aspect=Sim['Ndof']/(2*Sim['NBack']/Sim['nplt']),origin='lower',cmap='jet', interpolation='nearest')
#plt.plot(np.ones(10)*Result['tt'][-NBack], np.linspace(np.amin(Result['qoiTot']), np.amax(Result['qoiTot']), 10),'--',linewidth=3,color='k')
#fig = plt.figure()
#plt.plot(Result['uu'][-2*NBack,:int(Sim['Ndof']/2)],linewidth=3,color='k')
#plt.plot(Result['uu'][-1,:int(Sim['Ndof']/2)],'--',linewidth=3,color='r')
#plt.plot(Result['uu'][-NBack,:int(Sim['Ndof']/2)],linewidth=3,color='b')
#plt.plot(Result['tt'], Result['qoiTot'],linewidth=3,color='k')
