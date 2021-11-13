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
    np.savez('KSDATA.npz',qoi=Result['qoiTot'][indexLim::24,:,:], 
                          uu=Result['uu'][indexLim::24,:,:],
                          tt=Result['tt'][indexLim::24])
    
