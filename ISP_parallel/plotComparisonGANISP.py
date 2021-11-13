import numpy as np
import sys
sys.path.append('util')
from plotsUtil import *


#def clipGain(gain,biais):
     


filename_GenKS = 'GANISP'
filename_ISPKS = 'KS_C_2.5_N_45_eps_0.1'
filename_ISPL96 = 'L96_C_0.0104_N_64_eps_0.871'
keys = ['bruteProb', 'bruteProbStd', 'ispProb', 'ispProbStd', 'relBiais', 'compGain']
trueCDF = np.load('CDF/CDF_KS_9000000.npz')
trueX = trueCDF['xCDF']
trueY = trueCDF['yCDF']
minLevel = 1.75
maxLevel = 2.75
nLevels = 80
levels = np.linspace(minLevel,maxLevel,nLevels)
trueYInterp = np.interp(levels,trueX,trueY)


#plt.plot(Sim['Levels'], meanISP,color='b',linewidth=3,label='ISP')
#plt.plot(Sim['Levels'], meanISP + stdISP,'--',color='b',linewidth=3)
#plt.plot(Sim['Levels'], meanISP - stdISP,'--',color='b',linewidth=3)
#plt.plot(trueX,         meanBrute,color='k',linewidth=3,label='Truth')
#plt.plot(trueX,         meanBrute + stdBrute,'--',color='k',linewidth=3)
#plt.plot(trueX,         meanBrute - stdBrute,'--',color='k',linewidth=3)

#fig = plt.figure()
#Abrute = np.load(filename_ISPKS + '.npz')
#plt.plot(levels,Abrute['bruteProb'], color='k', linewidth=3,label='MC')
#A = np.load(filename_ISPKS + '.npz')
#plt.plot(levels,A['ispProb'], color='b', linewidth=2,label='ISP KS')
#plt.plot(levels,A['ispProb']+2*A['ispProbStd'], '--', color='b', linewidth=2)
#plt.plot(levels,A['ispProb']-2*A['ispProbStd'], '--', color='b', linewidth=2)
#prettyLabels('Q','Prob(Q)',14)
#plotLegend()
#ax = plt.gca()
#ax.set_yscale('log')




fig = plt.figure()
#L96
A = np.load(filename_ISPL96 + '.npz')
gain = A['compGain']
gain[np.argwhere(A['relBiais']>0.7)[:,0]]=np.nan
plt.plot(trueYInterp,gain, color='gray', linewidth=3,label='Random cloning L96')
#KS
A = np.load(filename_ISPKS + '.npz')
gain = A['compGain']
gain[np.argwhere(A['relBiais']>0.7)[:,0]]=np.nan
plt.plot(trueYInterp,gain, '--',color='b', linewidth=3,label='Random cloning KS')
#GANISP
A = np.load(filename_GenKS + '.npz')
gain = A['compGain']
gain[np.argwhere(A['relBiais']>0.7)[:,0]]=np.nan
plt.plot(trueYInterp,gain, color='b', linewidth=3,label='GANISP KS')
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
#plt.ylim([5e-2,10])
#leg=plt.legend(prop={'family':'Times New Roman','size': 22,'weight':'bold' },loc='lower left')
#leg.get_frame().set_linewidth(2.0)
#leg.get_frame().set_edgecolor('k')
prettyLabels('P','Computational Gain',30)
plt.tight_layout()
plt.savefig('Figures/Comparison_gain.eps')
plt.savefig('Figures/Comparison_gain.png')











plt.show()
