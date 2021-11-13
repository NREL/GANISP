import numpy as np
import sys
sys.path.append('util')
from plotsUtil import *
import os

os.makedirs('Figures',exist_ok=True)

     


filename_GenKS = 'KS_C_2.5_N_45_eps_0.1'
keys = ['bruteProb', 'bruteProbStd', 'ispProb', 'ispProbStd', 'relBiais', 'compGain','meanCloneDiff','allmeanNumberKills']
trueCDF = np.load('CDF/CDF_KS_9000000.npz')
trueX = trueCDF['xCDF']
trueY = trueCDF['yCDF']
minLevel = 1.75
maxLevel = 2.75
nLevels = 80
levels = np.linspace(minLevel,maxLevel,nLevels)
trueYInterp = np.interp(levels,trueX,trueY)


fig = plt.figure()
Abrute = np.load(filename_GenKS + '.npz')
plt.plot(levels,Abrute['bruteProb'], color='k', linewidth=3,label='MC')
plt.plot(levels,Abrute['bruteProb']+Abrute['bruteProbStd'], '--', color='k', linewidth=3)
plt.plot(levels,Abrute['bruteProb']-Abrute['bruteProbStd'], '--', color='k', linewidth=3)
A = np.load(filename_GenKS + '.npz')
plt.plot(levels,A['ispProb'], color='b', linewidth=3, label='GANISP')
plt.plot(levels,A['ispProb']+A['ispProbStd'], '--', color='b', linewidth=3)
plt.plot(levels,A['ispProb']-A['ispProbStd'], '--', color='b', linewidth=3)
prettyLabels('Q','P',30)
plotLegend(25)
ax = plt.gca()
ax.set_yscale('log')
plt.savefig('Figures/prob_ganisp.eps')
plt.savefig('Figures/prob_ganisp.png')


fig = plt.figure()
A = np.load(filename_GenKS + '.npz')
t = np.linspace(3.25,148.75,46)
plt.plot(t,128*A['meanCloneDiff'], '-o', color='k', linewidth=3)
plt.plot(50*np.ones(10),128*np.linspace(0.05,0.09,10), '--', color='k', linewidth=3)
prettyLabels('t',r'$||\xi_{parent} - \xi_{offspring}||_2$',30)
plt.tight_layout()
plt.savefig('Figures/diff_ganispOpt.eps')
plt.savefig('Figures/diff_ganispOpt.png')


plt.show()
