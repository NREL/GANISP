import numpy as np

def l96StepRK2(u,Sim):
    utmp = u.copy()
    source = u[Sim['im'],:] * (u[Sim['ip'],:]  - u[Sim['im2'],:])    + Sim['R'] - u
    k1 = utmp + Sim['Timestep'] * 0.5 * source
    source = k1[Sim['im'],:] * (k1[Sim['ip'],:]  - k1[Sim['im2'],:])    + Sim['R'] - k1
    u = utmp + Sim['Timestep'] * source
    return u

def l96StepBackRK2(u,Sim):
    utmp = u.copy()
    source = -u[Sim['im'],:] * (u[Sim['ip'],:]  - u[Sim['im2'],:])    - Sim['R'] + u
    k1 = utmp + Sim['Timestep'] * 0.5 * source
    source = -k1[Sim['im'],:] * (k1[Sim['ip'],:]  - k1[Sim['im2'],:])    - Sim['R'] + k1
    u = utmp + Sim['Timestep'] * source
    return u

def l96qoi(u):
    return np.sum(u**2)/(2*len(u))


def ksStepETDRK4(u,Sim):
    g = Sim['g']
    E = Sim['E']
    E_2 = Sim['E_2']
    Q = Sim['Q']
    f1 = Sim['f1']
    f2 = Sim['f2']
    f3 = Sim['f3']


    v = np.fft.fft(u,axis=0)

    Nv = g*np.fft.fft(np.real(np.fft.ifft(v,axis=0))**2,axis=0)
    a = E_2*v + Q*Nv
    Na = g*np.fft.fft(np.real(np.fft.ifft(a,axis=0))**2,axis=0)
    b = E_2*v + Q*Na
    Nb = g*np.fft.fft(np.real(np.fft.ifft(b,axis=0))**2,axis=0)
    c = E_2*a + Q*(2*Nb-Nv)
    Nc = g*np.fft.fft(np.real(np.fft.ifft(c,axis=0))**2,axis=0)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3

    u = np.real(np.fft.ifft(v,axis=0))

    return u

def ksStepRK2(u,Sim):
    g = Sim['g']
    E = Sim['E']
    E_2 = Sim['E_2']
    Q = Sim['Q']
    f1 = Sim['f1']
    f2 = Sim['f2']
    f3 = Sim['f3']
    k = Sim['k']
    h = Sim['Timestep']
    indexAlias = Sim['indexAlias']


    v = np.fft.fft(u,axis=0)
    vcopy = v.copy()

    v[indexAlias]=0
    source = (k**2-k**4)*v  - 0.5j*k*np.fft.fft(np.fft.ifft(v,axis=0)**2,axis=0)
    k1 = vcopy + h * 0.5 * source
    k1[indexAlias]=0
    source = (k**2-k**4)*k1  - 0.5j*k*np.fft.fft(np.fft.ifft(k1,axis=0)**2,axis=0)
    v = vcopy + h * source
     
    u = np.real(np.fft.ifft(v,axis=0))

    return u


def ksStepBackRegularizedETDRK4(u,Sim):
    beta = Sim['beta']
    k = Sim['k']
    h = Sim['Timestep']
    Ndof = Sim['Ndof']
    g = Sim['g']
    E = Sim['Eback']
    E_2 = Sim['E_2back']
    Q = Sim['Qback']
    f1 = Sim['f1back']
    f2 = Sim['f2back']
    f3 = Sim['f3back']
    indexAlias = Sim['indexAlias']
    dealias = False

    v = np.fft.fft(u,axis=0)
    

    if dealias:
        vtmp = v.copy()
        vtmp[indexAlias] = 0
        Nv = g*np.fft.fft(np.real(np.fft.ifft(vtmp,axis=0))**2,axis=0)
        a = E_2*v + Q*Nv
        atmp = a.copy()
        atmp[indexAlias] = 0
        Na = g*np.fft.fft(np.real(np.fft.ifft(atmp,axis=0))**2,axis=0)
        b = E_2*v + Q*Na
        btmp = b.copy()
        btmp[indexAlias] = 0
        Nb = g*np.fft.fft(np.real(np.fft.ifft(btmp,axis=0))**2,axis=0)
        c = E_2*a + Q*(2*Nb-Nv)
        ctmp = c.copy()
        ctmp[indexAlias] = 0
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c,axis=0))**2,axis=0)
    else:
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v,axis=0))**2,axis=0)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a,axis=0))**2,axis=0)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b,axis=0))**2,axis=0)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c,axis=0))**2,axis=0)
    
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    u = np.real(np.fft.ifft(v,axis=0))
    
    return u

def ksStepBackRegularizedRK2(u,Sim):
    beta = Sim['beta']
    k = Sim['k']
    h = Sim['Timestep']
    Ndof = Sim['Ndof']
    indexAlias = Sim['indexAlias']
   
    v = np.fft.fft(u,axis=0)
    vcopy = v.copy()
    v[indexAlias]=0
    source = -(k**2-k**4)*v/(1+beta*(k**4))  + 0.5j*k*np.fft.fft(np.fft.ifft(v,axis=0)**2,axis=0)/(1+beta*(k**4))
    k1 = vcopy + h * 0.5 * source
    k1[indexAlias]=0
    source = -(k**2-k**4)*k1/(1+beta*(k**4))  + 0.5j*k*np.fft.fft(np.fft.ifft(k1,axis=0)**2,axis=0)/(1+beta*(k**4))
    v = vcopy + h * source
     
    u = np.real(np.fft.ifft(v,axis=0))

    return u

