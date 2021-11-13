import sys
sys.path.append('../util')
import numpy as np
from plotsUtil import *
import myparser as myparser
import simulation as simulation
import data as data
import time
import monitorTiming as monitorTiming

# ~~~~ Init
# Parse input
inpt = myparser.parseInputFile()

# Initialize MPI
# Not for now

# Initialize random seed
np.random.seed(seed=42)

# Initialize Simulation details
Sim = data.simSetUp(inpt)

# Run Simulation
Result = simulation.simRunLE(Sim)

fig=plt.figure()
plt.plot(Result['LETime'],Result['LE'],color='k',linewidth=3)
prettyLabels('t',r'$\lambda$',14,title='Lyap. Exp.')

fig=plt.figure()
plt.plot(Result['LETime'],Result['LEQ'],color='k',linewidth=3)
prettyLabels('t',r'$\lambda$',14,title='Lyap. Exp. Q')

fig=plt.figure()
plt.plot(Result['LETime'],Result['LERunAve'],color='k',linewidth=3)
prettyLabels('t',r'$\lambda$',14,title='Lyap. Exp. = %.3f' % Result['LERunAve'][-1])

fig=plt.figure()
plt.plot(Result['LETime'],Result['LEQRunAve'],color='k',linewidth=3)
prettyLabels('t',r'$\lambda$',14,title='Lyap. Exp. Q = %.3f' % Result['LEQRunAve'][-1]  )


plt.show()



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
