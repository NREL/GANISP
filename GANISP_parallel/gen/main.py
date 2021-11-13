from gen_itfc import loadGen, cloneSample
import matplotlib.pyplot as plt
import numpy as np

def qoiFn(u):
    mean = np.mean(u)
    qoi = np.mean((u-mean)**2)
    return qoi

generator = loadGen()
qoiVal = 2.5
nClone = 2
genData = cloneSample(generator,qoiVal,nClone)


print(genData.shape)
print(qoiFn(genData[0,:,0]))
print(qoiFn(genData[1,:,0]))
diff = abs(genData[1,:,0]-genData[0,:,0])
print("diff =",np.mean(diff))
print("diff std = ",np.linalg.norm(diff))
print("max diff = ",np.linalg.norm(genData[0,:,0]))

plt.plot(genData[0,:,0])
plt.plot(genData[1,:,0])
plt.show()



