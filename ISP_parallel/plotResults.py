import numpy as np
import sys
sys.path.append('util')
from plotsUtil import *


#def clipGain(gain,biais):
     


filename_ISPKS = 'KS_C_2.5_N_45_eps_0.1'
filename_ISPL96 = 'L96_C_0.0104_N_64_eps_0.871'
keys = ['bruteProb', 'bruteProbStd', 'ispProb', 'ispProbStd', 'relBiais', 'compGain']

trueCDF_KS = np.load('CDF/CDF_KS_9000000.npz')
trueX_KS = trueCDF_KS['xCDF']
trueY_KS = trueCDF_KS['yCDF']
minLevel_KS = 1.75
maxLevel_KS = 2.75
nLevels_KS = 80
levels_KS = np.linspace(minLevel_KS,maxLevel_KS,nLevels_KS)
trueYInterp_KS = np.interp(levels_KS,trueX_KS,trueY_KS)
trueCDF_L96 = np.load('CDF/CDF_L96_10000000.npz')
trueX_L96 = trueCDF_L96['xCDF']
trueY_L96 = trueCDF_L96['yCDF']
minLevel_L96 = 1000
maxLevel_L96 = 1900
nLevels_L96 = 80
levels_L96 = np.linspace(minLevel_L96,maxLevel_L96,nLevels_L96)
trueYInterp_L96 = np.interp(levels_L96,trueX_L96,trueY_L96)


#plt.plot(Sim['Levels'], meanISP,color='b',linewidth=3,label='ISP')
#plt.plot(Sim['Levels'], meanISP + stdISP,'--',color='b',linewidth=3)
#plt.plot(Sim['Levels'], meanISP - stdISP,'--',color='b',linewidth=3)
#plt.plot(trueX,         meanBrute,color='k',linewidth=3,label='Truth')
#plt.plot(trueX,         meanBrute + stdBrute,'--',color='k',linewidth=3)
#plt.plot(trueX,         meanBrute - stdBrute,'--',color='k',linewidth=3)

fig = plt.figure()
Abrute = np.load(filename_ISPKS + '.npz')
plt.plot(levels_KS,Abrute['bruteProb'], color='k', linewidth=3,label='MC')
plt.plot(levels_KS,Abrute['bruteProb']+Abrute['bruteProbStd'], '--',color='k', linewidth=3)
plt.plot(levels_KS,Abrute['bruteProb']-Abrute['bruteProbStd'], '--', color='k', linewidth=3)
A = np.load(filename_ISPKS + '.npz')
plt.plot(levels_KS,A['ispProb'], color='b', linewidth=3,label='GAMS random cloning')
plt.plot(levels_KS,A['ispProb']+A['ispProbStd'], '--', color='b', linewidth=2)
plt.plot(levels_KS,A['ispProb']-A['ispProbStd'], '--', color='b', linewidth=2)
prettyLabels('Q','P',30,'Kuramoto-Sivashinsky')
#plotLegend(25)
ax = plt.gca()
ax.set_yscale('log')
#plt.ylim([1e-6,1])
fig.tight_layout()
plt.savefig('Figures/prob_ks.png')
plt.savefig('Figures/prob_ks.eps')

fig = plt.figure()
Abrute = np.load(filename_ISPL96 + '.npz')
plt.plot(levels_L96,Abrute['bruteProb'], color='k', linewidth=3,label='MC')
plt.plot(levels_L96,Abrute['bruteProb']+Abrute['bruteProbStd'], '--',color='k', linewidth=3)
plt.plot(levels_L96,Abrute['bruteProb']-Abrute['bruteProbStd'], '--',color='k', linewidth=3)
A = np.load(filename_ISPL96 + '.npz')
plt.plot(levels_L96,A['ispProb'], color='b', linewidth=3,label='GAMS random cloning')
plt.plot(levels_L96,A['ispProb']+A['ispProbStd'], '--', color='b', linewidth=2)
plt.plot(levels_L96,A['ispProb']-A['ispProbStd'], '--', color='b', linewidth=2)
prettyLabels('Q','P',30,'Lorenz 96')
#plotLegend(25)
ax = plt.gca()
ax.set_yscale('log')
#plt.ylim([1e-6,1])
fig.tight_layout()
plt.savefig('Figures/prob_l96.png')
plt.savefig('Figures/prob_l96.eps')














plt.show()
