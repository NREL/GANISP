import numpy as np
from integrator import *
from qoi import *
import parallel as par

def simSetUp(inpt):
    Sim = {}

    Ndof = int(inpt['Ndof'])
    Timestep = float(inpt['Timestep'])
    Tf = float(inpt['Tf'])
    NSim = int(inpt['NSim'])
    Sim['Simulation name'] = inpt['Simulation name']
    Sim['Ndof'] = Ndof
    Sim['Timestep'] = Timestep
    Sim['Tf'] = Tf
    Sim['NSim'] = NSim
    Sim['Record solution'] = (inpt['Record solution']=="True")

    # MPI parallelization
    nSim_, startSim_ = par.partitionSim(NSim)
    Sim['nSim_'] = nSim_
    Sim['startSim_'] = startSim_
    if par.nProc > 1:
        Sim['reconstruct Sol'] = (inpt['reconstruct Sol']=="True")
        Sim['reconstruct QOI'] = (inpt['reconstruct QOI']=="True")

    # Post proc
    Sim['Plot'] = (inpt['Plot']=="True")
    Sim['Build CDF'] = (inpt['Build CDF']=="True")   
    if Sim['Build CDF']:
       Sim['Plot CDF'] = (inpt['Plot CDF']=="True") 
    Sim['Build rare paths'] = (inpt['Build rare paths']=="True")   
    if Sim['Build rare paths']:
       Sim['Levels'] = [float(lev) for lev in inpt['Levels'].split()]
       Sim['Plot rare paths'] = (inpt['Plot rare paths']=="True") 

    if inpt['Simulation name'] == 'KS':
        # scalars for ETDRK4
        h = Timestep
        k = np.transpose(np.conj(np.concatenate((np.arange(0, Ndof/2.0), np.array([0]), np.arange(-Ndof/2.0+1.0, 0))))) / (float(inpt['Lx/pi'])/2.0)
        ksorted = list(abs(k))
        ksorted.sort()
        kalias = ksorted[int(len(ksorted)*2/3)]
        indexAlias = np.argwhere(abs(k)>kalias)
        L = k**2 - k**4
        E = np.exp(h*L)
        E_2 = np.exp(h*L/2)
        M = 16
        r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
        LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], Ndof, axis=0)
        Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
        f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
        f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
        f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
        tmax = Tf
        nmax = round(tmax/h)
        g = -0.5j*k

        # Necessary data for simulations         
        Sim['x'] = float(inpt['Lx/pi'])*np.pi*np.linspace(1,Ndof,Ndof)/Ndof
        Sim['E'] =   np.reshape(E,(Ndof,1))
        Sim['E_2'] = np.reshape(E_2,(Ndof,1))
        Sim['Q'] =   np.reshape(Q,(Ndof,1))
        Sim['f1'] =  np.reshape(f1,(Ndof,1))
        Sim['f2'] =  np.reshape(f2,(Ndof,1))
        Sim['f3'] =  np.reshape(f3,(Ndof,1))
        Sim['nmax'] = nmax
        Sim['nplt'] = 1
        Sim['g'] =   np.reshape(g,(Ndof,1))
        Sim['k'] =   np.reshape(k,(Ndof,1))
        Sim['indexAlias'] = indexAlias
        Sim['epsilon_init'] = float(inpt['epsilon_init'])

        # forward step and qoi
        Sim['stepFunc'] = ksStepETDRK4
        Sim['qoiFunc'] = ksqoi

        # Initial conditions
        ICType = inpt['ICType']
        if ICType=='file':
            fileNameIC = inpt['fileNameIC'] 
            Sim['u0'] = np.load(fileNameIC)
        elif ICType=='default':
            x = Sim['x']
            Sim['u0'] = np.cos(x/16)*(1+np.sin(x/16))
        else :
            print('IC type not recognized')

    if inpt['Simulation name'] == 'KSFrontBack':
        # scalars for ETDRK4
        h = Timestep
        k = np.transpose(np.conj(np.concatenate((np.arange(0, Ndof/2.0), np.array([0]), np.arange(-Ndof/2.0+1.0, 0))))) / (float(inpt['Lx/pi'])/2.0)
        ksorted = list(abs(k))
        ksorted.sort()
        kalias = ksorted[int(len(ksorted)*2/3)]
        indexAlias = np.argwhere(abs(k)>kalias)
        L = k**2 - k**4
        E = np.exp(h*L)
        E_2 = np.exp(h*L/2)
        M = 16
        r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
        LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], Ndof, axis=0)
        Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
        f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
        f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
        f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
        tmax = Tf
        nmax = round(tmax/h)
        g = -0.5j*k
        Sim['x'] = float(inpt['Lx/pi'])*np.pi*np.linspace(1,Ndof,Ndof)/Ndof
        Sim['E'] =   np.reshape(E,(Ndof,1))
        Sim['E_2'] = np.reshape(E_2,(Ndof,1))
        Sim['Q'] =   np.reshape(Q,(Ndof,1))
        Sim['f1'] =  np.reshape(f1,(Ndof,1))
        Sim['f2'] =  np.reshape(f2,(Ndof,1))
        Sim['f3'] =  np.reshape(f3,(Ndof,1))
        Sim['nmax'] = nmax
        Sim['nplt'] = 1
        Sim['g'] =   np.reshape(g,(Ndof,1))
        Sim['k'] =   np.reshape(k,(Ndof,1))
        Sim['indexAlias'] = indexAlias

        # Necessary data for simulations  
        beta = float(inpt['beta'])
        Lback = (k**2 - k**4)/(1+beta*k**4)
        Eback = np.exp(-h*Lback)
        E_2back = np.exp(-h*Lback/2)
        LRback = -h*np.transpose(np.repeat([Lback], M, axis=0)) + np.repeat([r], Ndof, axis=0)
        Qback = -h*np.real(np.mean((np.exp(LRback/2)-1)/LRback, axis=1))
        f1back = -h*np.real(np.mean((-4-LRback+np.exp(LRback)*(4-3*LRback+LRback**2))/LRback**3, axis=1)/(1+beta*k**4))
        f2back = -h*np.real(np.mean((2+LRback+np.exp(LRback)*(-2+LRback))/LRback**3, axis=1)/(1+beta*k**4))
        f3back = -h*np.real(np.mean((-4-3*LRback-LRback**2+np.exp(LRback)*(4-LRback))/LRback**3, axis=1)/(1+beta*k**4))
        g = -0.5j*k
        Sim['Eback'] =   np.reshape(Eback,(Ndof,1))
        Sim['E_2back'] = np.reshape(E_2back,(Ndof,1))
        Sim['Qback'] =   np.reshape(Qback,(Ndof,1))
        Sim['f1back'] =  np.reshape(f1back,(Ndof,1))
        Sim['f2back'] =  np.reshape(f2back,(Ndof,1))
        Sim['f3back'] =  np.reshape(f3back,(Ndof,1))
        Sim['beta'] = float(inpt['beta'])

        # forward step and qoi
        Sim['forwardStepFunc'] = ksStepETDRK4
        Sim['backwardStepFunc'] = ksStepBackRegularizedETDRK4
        Sim['qoiFunc'] = ksqoi

        # Initial conditions
        Sim['epsilon_init'] = float(inpt['epsilon_init'])
        ICType = inpt['ICType']
        if ICType=='file':
            fileNameIC = inpt['fileNameIC'] 
            Sim['u0'] = np.load(fileNameIC)
        elif ICType=='default':
            Sim['u0'] = np.cos(x/16)*(1+np.sin(x/16))
        else :
            print('IC type not recognized')

       
        # Initial conditions
        ICType = inpt['ICType']
        if ICType=='file':
            fileNameIC = inpt['fileNameIC'] 
            Sim['u0'] = np.load(fileNameIC)
        elif ICType=='default':
            x = Sim['x']
            Sim['u0'] = np.cos(x/16)*(1+np.sin(x/16))
        else :
            print('IC type not recognized')

    if inpt['Simulation name'] == 'L96':
        tmax = Tf
        nmax = round(tmax/Timestep)
        # Initial condition and grid setup
        epsilon_init = float(inpt['epsilon_init'])
        R = float(inpt['R L96'])
        im = np.zeros(Ndof)
        im2 = np.zeros(Ndof)
        ip = np.zeros(Ndof)
     
        ind = np.array(list(range(Ndof)))
      
        im = np.roll(ind,1)
        im2 = np.roll(ind,2)
        ip = np.roll(ind,-1)
        
        # Necessary data for simulations         
        Sim['epsilon_init'] = epsilon_init
        Sim['R'] = R
        Sim['im'] = im
        Sim['im2'] = im2
        Sim['ip'] = ip
        Sim['nmax'] = nmax
        Sim['nplt'] = 1


        # forward step and qoi
        Sim['stepFunc'] = l96StepRK2
        Sim['qoiFunc'] = l96qoi

        # Initial conditions
        ICType = inpt['ICType']
        if ICType=='file':
            fileNameIC = inpt['fileNameIC']
            Sim['u0'] = np.load(fileNameIC)
        elif ICType=='default':
            Sim['u0'] = np.zeros(Ndof)
        else :
            print('IC type not recognized') 



    if inpt['Simulation name'] == 'L96FrontBack':
        tmax = Tf
        nmax = round(tmax/Timestep)
        # Initial condition and grid setup
        epsilon_init = float(inpt['epsilon_init'])
        R = float(inpt['R L96'])
        im = np.zeros(Ndof)
        im2 = np.zeros(Ndof)
        ip = np.zeros(Ndof)
     
        ind = np.array(list(range(Ndof)))
      
        im = np.roll(ind,1)
        im2 = np.roll(ind,2)
        ip = np.roll(ind,-1)
        
        # Necessary data for simulations         
        Sim['epsilon_init'] = epsilon_init
        Sim['R'] = R
        Sim['im'] = im
        Sim['im2'] = im2
        Sim['ip'] = ip
        Sim['nmax'] = nmax
        Sim['nplt'] = 1

        # forward step and qoi
        Sim['forwardStepFunc'] = l96StepRK2
        Sim['backwardStepFunc'] = l96StepBackRK2
        Sim['qoiFunc'] = l96qoi

        # Initial conditions
        ICType = inpt['ICType']
        if ICType=='file':
            fileNameIC = inpt['fileNameIC']
            Sim['u0'] = np.load(fileNameIC)
        elif ICType=='default':
            Sim['u0'] = np.zeros(Ndof)
        else :
            print('IC type not recognized') 


    return Sim


