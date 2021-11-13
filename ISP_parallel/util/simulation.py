import numpy as np
import time
import sys
sys.path.append('util')
import importanceSplitting as isplt
from myProgressBar import printProgressBar

def run(Sim):
    # Timing
    t_start = time.time()
   
    t_init_start = time.time()

    Result = {}

    # Sim details
    Ndof = Sim['Ndof']
    NSim = Sim['NSim']
    h = Sim['Timestep']
    tmax = Sim['Tf']
    nmax = Sim['nmax']
    nplt = Sim['nplt']
    stepFunc = Sim['stepFunc']
    qoiFunc = Sim['qoiFunc']
    epsilon_init = Sim['epsilon_init']
    recordSolution = Sim['Record solution']
    nMonitor = int(round(nmax/100)) 


    uu = Sim['uu']
    tt = Sim['tt']

    # Initial condition
    u = np.transpose(Sim['u0']*np.ones((NSim,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,NSim))

    t_init_end = time.time()
    
    # main loop
    t_main_start = time.time()
    t_step = 0
    if recordSolution:
        uu[0,:,:]=u
    else:
        uu[0,:,0]=u[:,0]
    qoi = qoiFunc(u)
    Sim['qoiTot'][0,:,:]=qoi
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
            Sim['qoiTot'][n,:,:]=qoi
        if n%nMonitor == 0:
            advancementCounter +=1
        printProgressBar(n, nmax, prefix = 'Iter ' + str(n) + ' / ' +str(nmax),suffix = 'Complete', length = 50) 
    t_main_end = time.time()

    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = Sim['qoiTot']
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step

    return Result


def runEpsClone(Sim):
    # Timing
    t_start = time.time()
   
    t_init_start = time.time()

    Result = {}

    # Sim details
    Ndof = Sim['Ndof']
    NSim = Sim['NSim']
    h = Sim['Timestep']
    tmax = Sim['Tf']
    nmax = Sim['nmax']
    nplt = Sim['nplt']
    stepFunc = Sim['stepFunc']
    qoiFunc = Sim['qoiFunc']
    epsilon_init = Sim['epsilon_init']
    epsilon_clone = Sim['Epsilon clone']
    recordSolution = Sim['Record solution']
    nMonitor = int(round(nmax/100)) 


    uu = Sim['uu']
    tt = Sim['tt']

    # Initial condition
    u = np.transpose(Sim['u0']*np.ones((NSim,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,NSim))

    t_init_end = time.time()
    
    # main loop
    t_main_start = time.time()
    t_step = 0
    if recordSolution:
        uu[0,:,:]=u
    else:
        uu[0,:,0]=u[:,0]
    qoi = qoiFunc(u)
    Sim['qoiTot'][0,:,:]=qoi
    advancementCounter = 0
    printProgressBar(0, nmax, prefix = 'Iter ' + str(0) + ' / ' +str(nmax),suffix = 'Complete', length = 50) 
    for n in range(1, nmax+1):
        Sim['Ni'] += 1
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
            Sim['qoiTot'][n,:,:]=qoi
        if n%nMonitor == 0:
            advancementCounter +=1
        if  Sim['Ni'] == Sim['NselectionThreshold']:
            Sim['Ni'] = 0
            u += epsilon_clone*np.random.normal(loc=0.0,scale=1.0,size=u.shape) 
 
        printProgressBar(n, nmax, prefix = 'Iter ' + str(n) + ' / ' +str(nmax),suffix = 'Complete', length = 50) 
    t_main_end = time.time()

    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = Sim['qoiTot']
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step

    return Result



def runLE(Sim):
    # Timing
    t_start = time.time()
   
    t_init_start = time.time()

    Result = {}

    # Sim details
    Ndof = Sim['Ndof']
    NSim = 2
    NLE = NSim - 1
    h = Sim['Timestep']
    tmax = Sim['Tf']
    nmax = Sim['nmax']
    nplt = Sim['nplt']
    stepFunc = Sim['stepFunc']
    qoiFunc = Sim['qoiFunc']
    epsilon_init = Sim['epsilon_init']
    recordSolution = Sim['Record solution']
    nMonitor = int(round(nmax/100)) 
    normInit = Sim['normPerturb']
    nTimestepLE = 5


    uu = Sim['uu']
    tt = Sim['tt']

    # Initial condition
    u = np.transpose(Sim['u0']*np.ones((NSim,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,NSim))
    # Initialize perturbation
    pert = np.random.uniform(-0.5,0.5)*normInit
    # Normalize perturbation
    norm = np.linalg.norm(pert)
    # Apply perturbation
    u[:,1] = u[:,0] + pert*normInit/norm
     
    t_init_end = time.time()
    
    # main loop
    t_main_start = time.time()
    t_step = 0
    if recordSolution:
        uu[0,:,:]=u
    else:
        uu[0,:,0]=u[:,0]
    qoi = qoiFunc(u)
    Sim['qoiTot'][0,:,:]=qoi

    
    LEval = [] 
    LEQval = []
    LETime = []

    advancementLE = 0
    advancementCounter = 0
    printProgressBar(0, nmax, prefix = 'Iter ' + str(0) + ' / ' +str(nmax),suffix = 'Complete', length = 50)
    for n in range(1, nmax+1):
        if advancementLE==0:
            normQinit = abs(qoiFunc(u[:,0])-qoiFunc(u[:,1]))
        advancementLE += 1
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
            Sim['qoiTot'][n,:,:]=qoi
        if n%nMonitor == 0:
            advancementCounter +=1
        printProgressBar(n, nmax, prefix = 'Iter ' + str(n) + ' / ' +str(nmax),suffix = 'Complete', length = 50)
        if advancementLE>=nTimestepLE:
            normQDiff = abs(qoiFunc(u[:,0])-qoiFunc(u[:,1]))
            LEQval.append( (np.log(normQDiff)-np.log(normQinit))/(nTimestepLE*h) )
            diff = u[:,1]-u[:,0]
            normDiff = np.linalg.norm(diff)
            LEval.append( (np.log(normDiff)-np.log(normInit))/(nTimestepLE*h) )
            u[:,1] = u[:,0] + diff*normInit/normDiff
            advancementLE=0
            LETime.append((n-advancementLE)*h)
 
    t_main_end = time.time()

    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = Sim['qoiTot']
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step
    Result['LE'] = np.array(LEval)
    cumulativeLE = np.cumsum(Result['LE'])
    Result['LERunAve'] = cumulativeLE/np.array(list(range(1,len(cumulativeLE)+1)))
    Result['LEQ'] = np.array(LEQval)
    cumulativeLEQ = np.cumsum(Result['LEQ'])
    Result['LEQRunAve'] = cumulativeLEQ/np.array(list(range(1,len(cumulativeLEQ)+1)))
    Result['LETime'] = np.array(LETime)


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
    recordSolution = Sim['Record solution']

    uu = Sim['uu']
    tt = Sim['tt']
  
    # Initial conditions
    u = np.transpose(Sim['u0']*np.ones((NSim,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,NSim))

    t_init_end = time.time()

    # main loop
    t_main_start = time.time()
    if recordSolution:
        uu[0,:,:]=u
    else:
        uu[0,:,0]=u[:,0]
    qoi = qoiFunc(u)
    Sim['qoiTot'][0,:,:] = qoi
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
            Sim['qoiTot'][n,:,:] = qoi
    t_main_end = time.time()
   
    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = Sim['qoiTot']
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step

    return Result

def runIS(Sim):
    # Timing
    t_start = time.time()
   
    t_init_start = time.time()

    Result = {}

    # Sim details
    Ndof = Sim['Ndof']
    NSim = Sim['NSim']
    h = Sim['Timestep']
    tmax = Sim['Tf']
    nmax = Sim['nmax']
    nplt = Sim['nplt']
    stepFunc = Sim['stepFunc']
    qoiFunc = Sim['qoiFunc']
    epsilon_init = Sim['epsilon_init']
    recordSolution = Sim['Record solution']
    nMonitor = int(round(nmax/100)) 

    uu = Sim['uu']
    tt = Sim['tt']


    t_init_end = time.time()
   
    printProgressBar(0, Sim['nRep_'], prefix = 'ISP reps ' + str(0) + ' / ' +str(Sim['nRep_']),suffix = 'Complete', length = 50) 
    for iRep in range(Sim['nRep_']):
        # main loop
        t_main_start = time.time()
        isplt.reset(Sim)
        t_step = 0
        # Initial condition
        u = np.transpose(Sim['u0']*np.ones((NSim,Ndof))) + epsilon_init*np.random.normal(loc=0.0, scale=1.0, size=(Ndof,NSim))
        if recordSolution:
            uu[0,:,:]=u
        else:
            uu[0,:,0]=u[:,0]
        qoi = qoiFunc(u)
        Sim['qoiTot'][0,:,:]=qoi
        advancementCounter = 0
        for n in range(1, nmax+1):
            isplt.prestep(u,Sim,n)
            t = n*h
            t_start_step = time.time()
            u = stepFunc(u,Sim)
            isplt.step(Sim)
            t_end_step = time.time()
            t_step += t_end_step-t_start_step
            qoi = qoiFunc(u) 
            Sim['qoiTot'][n,:,:]=qoi
            Sim['u'] = u
            isplt.poststep(Sim,n,iRep)
            u = Sim['u']
            if n%nplt == 0:
                if recordSolution:
                    uu[n,:,:] = u
                else:
                    uu[n,:,0] = u[:,0]
            if n%nMonitor == 0:
                advancementCounter +=1
                #print("Done " + str(round(100*float(n/(nmax)))) + "%")

        # ~~~~ Finalize
        isplt.finalize(Sim,iRep)

        # ~~~~ Log advancement
        printProgressBar(iRep+1, Sim['nRep_'], prefix = 'ISP reps ' + str(iRep+1) + ' / ' +str(Sim['nRep_']),suffix = 'Complete', length = 50) 

        t_main_end = time.time()

    # Timing
    t_end = time.time()

    Result['tt'] = tt
    Result['uu'] = uu
    Result['qoiTot'] = Sim['qoiTot']
    Result['timeExec'] = t_end-t_start
    Result['timeExecInit'] = t_init_end-t_init_start
    Result['timeExecMain'] = t_main_end-t_main_start
    Result['timeExecStep'] = t_step

    return Result

def simRun(Sim):
    
    if Sim['Simulation name']=='KS' or Sim['Simulation name']=='L96':
        return run(Sim)
    if Sim['Simulation name']=='KSFrontBack' or Sim['Simulation name']=='L96FrontBack':
        return runFrontBack(Sim)

def simRunLE(Sim):
    
    if Sim['Simulation name']=='KS' or Sim['Simulation name']=='L96':
        return runLE(Sim)
    if Sim['Simulation name']=='KSFrontBack' or Sim['Simulation name']=='L96FrontBack':
        print('not implemented, LE is done only for forward sim')
        sys.exit()
        return 


def simRunEpsClone(Sim):
    
    if Sim['Simulation name']=='KS' or Sim['Simulation name']=='L96':
        return runEpsClone(Sim)
    if Sim['Simulation name']=='KSFrontBack' or Sim['Simulation name']=='L96FrontBack':
        print('not implemented, LE is done only for forward sim')
        sys.exit()
        return 



def simRunIS(Sim):
    
    if Sim['Simulation name']=='KS' or Sim['Simulation name']=='L96':
        return runIS(Sim)
