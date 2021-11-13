import numpy as np
import parallel as par
import time
from myProgressBar import printProgressBar


def run(Sim):
    # Timing
    t_start = time.time()
   
    t_init_start = time.time()

    Result = {}

    Ndof = Sim['Ndof']
    NSim = Sim['NSim']
    h = Sim['Timestep']
    tmax = Sim['Tf']
    nmax = Sim['nmax']
    nplt = Sim['nplt']
    stepFunc = Sim['stepFunc']
    qoiFunc = Sim['qoiFunc']
    epsilon_init = Sim['epsilon_init']
    nSim_ = Sim['nSim_']
    recordSolution = Sim['Record solution']
    nMonitor = int(nmax/100) 
    buildRarePaths = Sim['Build rare paths']
    if buildRarePaths:
        rareLevels = Sim['Levels']

    if recordSolution:
        uu = np.zeros((nmax+1,Ndof,nSim_))
    else :
        uu = np.zeros((nmax+1,Ndof,1))
    tt = np.arange(0,(nmax+1)*h,h)
    qoiTot = np.zeros((nmax+1,1,nSim_))


    u = np.transpose(Sim['u0']*np.ones((nSim_,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,nSim_))

    t_init_end = time.time()
    
    # main loop
    t_main_start = time.time()
    t_step = 0
    if recordSolution:
        uu[0,:,:]=u
    else:
        uu[0,:,0]=u[:,0]
    qoi = qoiFunc(u)
    qoiTot[0,:,:]=qoi
    advancementCounter = 0
    printProgressBar(0, nmax, prefix = 'Iter ' + str(0) + ' / ' +str(nmax),suffix = 'Complete', length = 50)
    for n in range(1, nmax+1):
        t = n*h
        t_start_step = time.time()
        u = stepFunc(u,Sim)
        t_end_step = time.time()
        t_step += t_end_step-t_start_step
        qoi = qoiFunc(u) 
        if n%nplt == 0:
            if recordSolution:
                uu[n,:,:] = u
            else:
                uu[n,:,0] = u[:,0]
            qoiTot[n,:,:]=qoi
        if n%nMonitor == 0:
            advancementCounter +=1
            #par.printRoot("Done " + str(advancementCounter) + "%")
        printProgressBar(n, nmax, prefix = 'Iter ' + str(n) + ' / ' +str(nmax),suffix = 'Complete', length = 50)
    t_main_end = time.time()

    # Reconstruct results
    t_recons_start = time.time()
    uu, qoiTot = par.reconstruct(uu,qoiTot,Sim)
    t_recons_end = time.time()
 
    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = qoiTot
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step
    Result['timeExecRecons'] = t_recons_end-t_recons_start

    return Result

def runFrontBack(Sim):
   
    # Timing
    t_start = time.time()

    t_init_start = time.time()
    Result = {}

    Ndof = Sim['Ndof']
    NSim = Sim['NSim']
    h = Sim['Timestep']
    tmax = Sim['Tf']
    nmax = Sim['nmax']
    nplt = Sim['nplt']
    stepFuncForward = Sim['forwardStepFunc']
    stepFuncBackward = Sim['backwardStepFunc']
    qoiFunc = Sim['qoiFunc']
    epsilon_init = Sim['epsilon_init']
    nSim_ = Sim['nSim_']
 
    recordSolution = Sim['Record solution']
 
    if recordSolution:
        uu = np.zeros((nmax+1,Ndof,nSim_))
    else :
        uu = np.zeros((nmax+1,Ndof,1))
    tt = np.arange(0,(nmax+1)*h,h)
    qoiTot = np.zeros((nmax+1,1,nSim_))

    u = np.transpose(Sim['u0']*np.ones((nSim_,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,nSim_))

    t_init_end = time.time()

    # main loop
    t_main_start = time.time()
    if recordSolution:
        uu[0,:,:]=u
    else:
        uu[0,:,0]=u[:,0]
    qoi = qoiFunc(u)
    qoiTot[0,:,:] = qoi
    t_step = 0
    for n in range(1, nmax+1):
        t = n*h
        t_start_step = time.time()
        if n<nmax/2:
            u = stepFuncForward(u,Sim)
        else:
            u = stepFuncBackward(u,Sim)
        t_end_step = time.time()
        t_step += t_end_step-t_start_step
        qoi = qoiFunc(u)    
        if n%nplt == 0:
            if recordSolution:
                uu[n,:,:] = u
            else:
                uu[n,:,0] = u[:,0]
            qoiTot[n,:,:] = qoi
    t_main_end = time.time()
   
    # Reconstruct results
    t_recons_start = time.time()
    uu, qoiTot = par.reconstruct(uu,qoiTot,Sim)
    t_recons_end = time.time()

    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = qoiTot
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step
    Result['timeExecRecons'] = t_recons_end-t_recons_start

    return Result

def simRun(Sim):
    
    if Sim['Simulation name']=='KS' or Sim['Simulation name']=='L96':
        return run(Sim)
    if Sim['Simulation name']=='KSFrontBack' or Sim['Simulation name']=='L96FrontBack':
        return runFrontBack(Sim)
