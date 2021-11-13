import numpy as np


def l96qoi(u):
    return np.sum(u**2,axis=0)/(2*len(u[:,0]))


def ksqoi(u):
    meanu = np.mean(u,axis=0)
    return np.sum((u-meanu)**2,axis=0)/(len(u[:,0]))
