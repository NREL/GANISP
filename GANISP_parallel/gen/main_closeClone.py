from gen_itfc import loadGen, cloneSample, closeCloneSample, recursiveCloseCloneSample
import matplotlib.pyplot as plt
import numpy as np

def qoiFn(u):
    mean = np.mean(u)
    qoi = np.mean((u-mean)**2)
    return qoi

data = np.load('test_data.npy')
uref = data[np.random.randint(0,1700),:]


generator = loadGen()
qoiVal = qoiFn(uref)
nClone = 2
genDataClose = closeCloneSample(generator,qoiVal,nClone,uref)
genDataRec = recursiveCloseCloneSample(generator,qoiVal,nClone,uref)
#genData = cloneSample(generator,qoiVal,nClone)

print(qoiFn(uref))
print(qoiFn(genDataClose[0,:,0]))
print(qoiFn(genDataRec[0,:,0]))
for i in range(nClone):
    plt.plot(genDataClose[0,:,0],label='close')
    plt.plot(genDataRec[0,:,0],label='Rec')
plt.plot(uref,linewidth=3)
plt.show()

plt.plot(uref,linewidth=3)
plt.plot(uref+0.3*np.random.normal(0,1,uref.shape))
plt.show()
