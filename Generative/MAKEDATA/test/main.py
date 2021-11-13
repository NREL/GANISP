import sys
sys.path.append('../util')
import numpy as np
from plotsUtil import *
import myparser as myparser
import simulation as simulation
import data as data
import parallel as par
import time
import monitorTiming as monitorTiming

# ~~~~ Init
# Parse input
inpt = myparser.parseInputFile()

# Initialize MPI
# Not for now

# Initialize random seed
if inpt['Simulation name']=='L96':
   ref = np.load('l96UnitTest.npz')
if inpt['Simulation name']=='L96FrontBack':
   ref = np.load('l96FrontBackUnitTest.npz')
if inpt['Simulation name']=='KS':
   ref = np.load('KSUnitTest.npz')
if inpt['Simulation name']=='KSFrontBack':
   ref = np.load('KSFrontBackUnitTest.npz')
   
np.random.seed(seed=ref['seed'])

# Initialize Simulation details
Sim = data.simSetUp(inpt)

# Run Simulation
Result = simulation.simRun(Sim)


smallNumber = 1e-16


def checkParam(ref,Sim,paramName,sameSim):
    if type(Sim[paramName]) is np.ndarray:
        if not np.amin(abs(Sim[paramName]-ref[paramName]))<smallNumber:
            sameSim = False
            print("Cannot complete test, "+paramName+" = ", str(Sim[paramName]), " instead of ", str(ref[paramName]))
    else:
        if not Sim[paramName]==ref[paramName]:
            sameSim = False
            print("Cannot complete test, "+paramName+" = ", str(Sim[paramName]), " instead of ", str(ref[paramName]))

def checkSolution(ref,Result,smallNumber):
    if np.amax(abs(np.squeeze(Result['tt'])-ref['tt']))>smallNumber or np.amax(abs(np.squeeze(Result['uu'])-ref['uu']))>smallNumber or np.amax(abs(np.squeeze(Result['qoiTot'])-ref['qoiTot']))>smallNumber:
        print(' ################## FAIL')
        print(np.squeeze(Result['tt'])-ref['tt'])
        print(ref['tt'])
        print(Result['tt'])
    else:
        print('PASS')



if Sim['Simulation name']=='L96':
   
   sameSim = True
   checkParam(ref,Sim,'Tf',sameSim)
   checkParam(ref,Sim,'R',sameSim)
   checkParam(ref,Sim,'Timestep',sameSim)
   checkParam(ref,Sim,'Ndof',sameSim)

   if sameSim==False:
      sys.exit()
   else:
      print("Checking ...")
 
   checkSolution(ref,Result,smallNumber)      
  
if Sim['Simulation name']=='L96FrontBack':
   
   sameSim = True
   checkParam(ref,Sim,'Tf',sameSim)
   checkParam(ref,Sim,'R',sameSim)
   checkParam(ref,Sim,'Timestep',sameSim)
   checkParam(ref,Sim,'Ndof',sameSim)
   checkParam(ref,Sim,'u0',sameSim)

   if sameSim==False:
      sys.exit()
   else:
      print("Checking ...")
 
   checkSolution(ref,Result,smallNumber)      

if Sim['Simulation name']=='KS':
   
   sameSim = True
   checkParam(ref,Sim,'Tf',sameSim)
   checkParam(ref,Sim,'Timestep',sameSim)
   checkParam(ref,Sim,'Ndof',sameSim)
   checkParam(ref,Sim,'u0',sameSim)

   if sameSim==False:
      sys.exit()
   else:
      print("Checking ...")
 
   checkSolution(ref,Result,smallNumber)      

if Sim['Simulation name']=='KSFrontBack':
   
   sameSim = True
   checkParam(ref,Sim,'Tf',sameSim)
   checkParam(ref,Sim,'Timestep',sameSim)
   checkParam(ref,Sim,'Ndof',sameSim)
   checkParam(ref,Sim,'beta',sameSim)
   checkParam(ref,Sim,'u0',sameSim)

   if sameSim==False:
      sys.exit()
   else:
      print("Checking ...")
 
   checkSolution(ref,Result,smallNumber)      


## Plot Result
#nmax = Sim['nmax']
#fig = plt.figure()
#plt.imshow(np.transpose(np.transpose(Result['uu'])),aspect=Sim['Ndof']/(Sim['nmax']/Sim['nplt']),origin='lower',cmap='jet', interpolation='nearest')
#prettyLabels('x','t',14)
#plt.colorbar()
#
#fig=plt.figure()
#plt.plot(Result['tt'], Result['qoiTot'],linewidth=3,color='k')
#plt.plot(np.ones(10)*Result['tt'][int(nmax/2)], np.linspace(np.amin(Result['qoiTot']), np.amax(Result['qoiTot']), 10),'--',linewidth=3,color='k')
#prettyLabels('T','E(u)',14)

#fig = plt.figure()
#plt.plot(Result['uu'][0,:],linewidth=3,color='k',label='start')
#plt.plot(Result['uu'][int(nmax/2),:],linewidth=3,color='b',label= str(int(nmax/2))+ r'$\Delta$ t')
#plt.plot(Result['uu'][-1,:],'--',linewidth=3,color='r',label=r'start recons')
#prettyLabels('x','u',14)
#plotLegend()

#if Sim['Simulation name']=='L96':
#   np.savez('l96UnitTest',tt=Result['tt'], uu=Result['uu'], qoiTot=Result['qoiTot'],Ndof=Sim['Ndof'],Timestep=Sim['Timestep'],R=Sim['R'],Tf=Sim['Tf'],seed=42)
#if Sim['Simulation name']=='KS':
#   np.savez('KSUnitTest',tt=Result['tt'], uu=Result['uu'], qoiTot=Result['qoiTot'],Ndof=Sim['Ndof'],Timestep=Sim['Timestep'],Tf=Sim['Tf'],u0=Sim['u0'], seed=42)
#if Sim['Simulation name']=='KSFrontBack':
#   np.savez('KSFrontBackUnitTest',tt=Result['tt'], uu=Result['uu'], qoiTot=Result['qoiTot'],Ndof=Sim['Ndof'],Timestep=Sim['Timestep'],Tf=Sim['Tf'],u0=Sim['u0'], \
#                                                                                            beta=Sim['beta'],seed=42)
#if Sim['Simulation name']=='L96FrontBack':
#   np.savez('L96FrontBackUnitTest',tt=Result['tt'], uu=Result['uu'], qoiTot=Result['qoiTot'],Ndof=Sim['Ndof'],Timestep=Sim['Timestep'],Tf=Sim['Tf'],u0=Sim['u0'],R=Sim['R'],seed=42)
