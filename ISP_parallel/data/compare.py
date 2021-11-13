import numpy as np
import sys
sys.path.append('../util')
from plotsUtil import *

exec(open('dataREWoutersBouchet.py').read())

M0_261 = np.load('Mean_C0.00261.npz')
M0_521 = np.load('Mean_C0.00521.npz')
M0_782 = np.load('Mean_C0.00782.npz')
M1_04 = np.load('Mean_C0.0104.npz')

NM0_261 = np.load('Nomean_C0.00261.npz')
NM0_521 = np.load('Nomean_C0.00521.npz')
NM0_782 = np.load('Nomean_C0.00782.npz')
NM1_04 = np.load('Nomean_C0.0104.npz')

fig=plt.figure()
plt.plot(XBF,YBF,'o',color='k',label='Ref')
plt.plot(M0_261['levels'],M0_261['bruteRe'],color='k',linewidth=3,label='Mean')
plt.plot(NM0_261['levels'],NM0_261['bruteRe'],'-+',color='k',linewidth=3,label='No Mean')
plt.plot(M0_261['levels'],M0_261['bruteRe'],color='k',linewidth=3,label='BruteForce')

plt.plot(X0_261,Y0_261,'o',color='g')
plt.plot(M0_261['levels'],M0_261['ispRe'],color='g',linewidth=3,label='C=0.00261')
plt.plot(NM0_261['levels'],NM0_261['ispRe'],'-+',color='g',linewidth=3)

plt.plot(X0_521,Y0_521,'o',color='r')
plt.plot(M0_521['levels'],M0_521['ispRe'],color='r',linewidth=3,label='C=0.00521')
plt.plot(NM0_521['levels'],NM0_521['ispRe'],'-+',color='r',linewidth=3)

plt.plot(X0_782,Y0_782,'o',color='b')
plt.plot(M0_782['levels'],M0_782['ispRe'],color='b',linewidth=3,label='C=0.00782')
plt.plot(NM0_782['levels'],NM0_782['ispRe'],'-+',color='b',linewidth=3)

plt.plot(X1_04,Y1_04,'o',color='c')
plt.plot(M1_04['levels'],M1_04['ispRe'],color='c',linewidth=3,label='C=0.0104')
plt.plot(NM1_04['levels'],NM1_04['ispRe'],'-+',color='c',linewidth=3)

plotLegend()
prettyLabels('a','RE',14)


fig=plt.figure()
plt.plot(XBF,YBF,'o',color='k',label='Ref')
plt.plot(NM0_261['levels'],NM0_261['bruteRe'],color='k',linewidth=3,label='Impl.')
plt.plot(NM0_261['levels'],NM0_261['bruteRe'],color='k',linewidth=3,label='BruteForce')

plt.plot(X0_261,Y0_261,'o',color='g')
plt.plot(NM0_261['levels'],NM0_261['ispRe'],color='g',linewidth=3,label='C=0.00261')

plt.plot(X0_521,Y0_521,'o',color='r')
plt.plot(NM0_521['levels'],NM0_521['ispRe'],color='r',linewidth=3,label='C=0.00521')

plt.plot(X0_782,Y0_782,'o',color='b')
plt.plot(NM0_782['levels'],NM0_782['ispRe'],color='b',linewidth=3,label='C=0.00782')

plt.plot(X1_04,Y1_04,'o',color='c')
plt.plot(NM1_04['levels'],NM1_04['ispRe'],color='c',linewidth=3,label='C=0.0104')

plotLegend()
prettyLabels('a','RE',14)




fig=plt.figure()
plt.plot(XBF,YBF,'o',color='k',label='Ref')
plt.plot(M0_261['levels'],M0_261['bruteRe'],color='k',linewidth=3,label='Impl.')
plt.plot(M0_261['levels'],M0_261['bruteRe'],color='k',linewidth=3,label='BruteForce')

plt.plot(X0_261,Y0_261,'o',color='g')
plt.plot(M0_261['levels'],M0_261['ispRe'],color='g',linewidth=3,label='C=0.00261')

plt.plot(X0_521,Y0_521,'o',color='r')
plt.plot(M0_521['levels'],M0_521['ispRe'],color='r',linewidth=3,label='C=0.00521')

plt.plot(X0_782,Y0_782,'o',color='b')
plt.plot(M0_782['levels'],M0_782['ispRe'],color='b',linewidth=3,label='C=0.00782')

plt.plot(X1_04,Y1_04,'o',color='c')
plt.plot(M1_04['levels'],M1_04['ispRe'],color='c',linewidth=3,label='C=0.0104')

plotLegend()
prettyLabels('a','RE',14)









plt.show()
