import numpy as np


def l96qoi(u):
    return np.sum(u**2,axis=0)/(2*u.shape[0])


def ksqoi(u):
    meanu = np.mean(u,axis=0)
    return np.sum((u-meanu)**2,axis=0)/(u.shape[0])


def qoiWrap(u,Sim):
    if Sim['Simulation name']=='L96':
        return l96qoi(u)
    elif Sim['Simulation name']=='KS':
        return ksqoi(u)
    
