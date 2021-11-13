import numpy as np
import sys
import qoi as qoi
sys.path.append('gen')
from gen_itfc import loadGen, cloneSample, closeCloneSample, recursiveCloseCloneSample 
import time
import parallel as par
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ! 
# ~~~~ Selection without replacement
# ~~~~ Sample K numbers from an array 0...N-1 and output them
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ !
def ransam(N,K):
    if N<K:
        print("ERROR in ransam: N = " + str(N) + " < K = "+str(K) )
        sys.exit()
    return np.random.choice(list(range(N)), size=K, replace=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ! 
# ~~~~ Selection with replacement
# ~~~~ Sample K numbers from an array 0...N-1 and output them
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ !
def ransam_rep(N,K):
    return np.random.choice(list(range(N)), size=K, replace=True)


def simSetUp(inpt,Sim):
    #np.random.seed(seed=Sim['Seed'])
    NSim = Sim['NSim']
    nmax = Sim['nmax']

    # MPI parallelization
    Sim['NRep'] = int(inpt['NRep'])
    nRep_, startRep_ = par.partitionSim(Sim['NRep'])
    Sim['nRep_'] = nRep_
    Sim['startRep_'] = startRep_     

    Sim['Target level'] = float(inpt['Target level'])
    Sim['Nselection'] = int(inpt['Nselection'])
    Sim['Cweight'] = float(inpt['Cweight'])
    Sim['Epsilon clone'] = float(inpt['Epsilon clone'])
    Sim['Min target level'] = float(inpt['Min target level'])
    Sim['Max target level'] = float(inpt['Max target level'])
    Sim['Number of thresholds'] = int(inpt['Number of thresholds'])
    Sim['NselectionThreshold'] = int(round((nmax+1)/Sim['Nselection'])) # Number of steps between cloning 
    Sim['NselectionTotal'] = int((nmax+1)/Sim['NselectionThreshold']) # Actual number of cloning 
    minLevel = Sim['Min target level']
    maxLevel = Sim['Max target level']
    nLevels = Sim['Number of thresholds']
    Sim['Levels'] = np.linspace(minLevel,maxLevel,nLevels)

    # Post proc
    Sim['Plot ISP CDF'] = (inpt['Plot ISP CDF']=='True') 
    if Sim['Plot ISP CDF']:
        Sim['True CDF file'] = inpt['True CDF file'] 
    Sim['Plot kill history'] = (inpt['Plot kill history']=='True')  
    try:
        Sim['Plot clone history'] = (inpt['Plot clone history']=='True')  
    except KeyError:
        Sim['Plot clone history'] = False

    par.printRoot('Asked for ' + str(Sim['Nselection']) + ' cloning steps')
    par.printRoot('Will perform ' + str(Sim['NselectionTotal']) + ' cloning steps')
    par.printRoot('Number of step between cloning ' + str(Sim['NselectionThreshold']))
 
    # Make sure the cloning step properly divide the simulation  
    if nmax+1-Sim['NselectionThreshold']*Sim['NselectionTotal'] > Sim['NselectionThreshold']:
        par.printRoot('ERROR: Problem in setup of number of cloning steps')
        sys.exit()

    if nmax+1-Sim['NselectionThreshold']*Sim['NselectionTotal'] < 5:
        par.printRoot('WARNING: last cloning will be done with ' + str(nmax+1-Sim['NselectionThreshold']*Sim['NselectionTotal']) +'steps')
   
    # Monitor  
    Sim['numberKills'] = np.zeros((Sim['NselectionTotal'],Sim['nRep_']))
    Sim['diffClone'] = np.zeros((Sim['NselectionTotal'],Sim['nRep_']))
    Sim['nClonesForDiff'] = np.zeros((Sim['NselectionTotal'],Sim['nRep_']))
    Sim['probabilities'] = np.zeros((Sim['Number of thresholds'],Sim['nRep_']))

    Sim['Elastclone'] = np.zeros(NSim)
    Sim['W'] = np.zeros((nmax+1,NSim))
    Sim['Wbar'] = np.zeros((nmax+1,NSim))
    Sim['Z'] = np.zeros(nmax+1)
    Sim['numberClones'] = np.zeros(NSim,dtype=int)
    Sim['nCloneAvail'] = 0
    
    # Useful markers
    Sim['Ni'] = 0 # Timestep counter to know when to clone
    Sim['NselectionLoc'] = 0 # how many times have we cloned
    Sim['timestepLastClone'] = 0 # when was the last cloning
  
    # For probabilities computation
    Sim['F_prob'] = np.zeros(NSim)
    Sim['Rw'] = np.zeros(Sim['NselectionTotal']+1)    

    # Rare path
    try:
        Sim['UseRarePath'] = (inpt['UseRarePath']=='True')
    except KeyError:
        Sim['UseRarePath'] = False
    if Sim['UseRarePath']:
        Sim['RarePathFile'] = inpt['RarePathFile']
        Sim['scaleFlucC'] = float(inpt['scaleFlucC'])
        Sim['meanPath'] = np.load(Sim['RarePathFile'])['meanPath']     
        Sim['varPath'] = (np.load(Sim['RarePathFile'])['stdPath'])**2
        Sim['rarePath'] = np.load(Sim['RarePathFile'])['rarePath']
        for i in range(len(Sim['varPath'])):
            if Sim['varPath'][i]<1e-6:
                Sim['varPath'][i] = np.amax(Sim['varPath'])
               
    # Load generator for cloning
    try:
        Sim['UseGAN'] = (inpt['UseGAN']=='True')
    except KeyError:
        Sim['UseGAN'] = False

    if (not Sim['Simulation name']=='KS') and Sim['UseGAN']:
        par.printRoot('GAN was trained only with KS so far')
        par.printRoot('Random cloning will be performed')
        Sim['UseGAN'] = False

    if Sim['UseGAN']:
        Sim['modelWeightFile'] = inpt['modelWeightFile']
        Sim['generator'] = loadGen(Sim['modelWeightFile'])


def reset(Sim):

    NSim = Sim['NSim']
    nmax = Sim['nmax']
    Sim['Elastclone'] = np.zeros(NSim)
    Sim['W'] = np.zeros((nmax+1,NSim))
    Sim['Wbar'] = np.zeros((nmax+1,NSim))
    Sim['Z'] = np.zeros(nmax+1)
    Sim['numberClones'] = np.zeros(NSim,dtype=int)
    Sim['nCloneAvail'] = 0
    
    # Useful markers
    Sim['Ni'] = 0 # Timestep counter to know when to clone
    Sim['NselectionLoc'] = 0 # how many times have we cloned
    Sim['timestepLastClone'] = 0 # when was the last cloning
  
    # For probabilities computation
    Sim['F_prob'] = np.zeros(NSim)
    Sim['Rw'] = np.zeros(Sim['NselectionTotal']+1)    

def computeRareFlucC(Sim,itime):
    return np.clip(Sim['scaleFlucC']*(Sim['rarePath'][itime] - Sim['meanPath'][itime])/Sim['varPath'][itime],0,1000)

def computeWeights(Sim,itime):
  
    qoi = Sim['qoiTot'][itime,0,:] 

    # Weights
    if not Sim['UseRarePath']:
        Sim['W'][itime,:] = np.exp(Sim['Cweight']*(qoi-Sim['Elastclone']))
    else:
        C = computeRareFlucC(Sim,itime)
        #print("C = ",C)
        ClastClone = computeRareFlucC(Sim,Sim['timestepLastClone'])
        Sim['W'][itime,:] = np.exp(C*qoi-ClastClone*Sim['Elastclone'])

    # Advance markers
    Sim['NselectionLoc'] += 1
    #print("cloning # " + str(Sim['NselectionLoc']))

    # Reinitialize timestep marker
    Sim['Ni'] = 0
  
    # Compute Z
    Sim['Z'][itime] = np.mean(Sim['W'][itime,:])

    # Initialize parameters for cloning
    rnd = np.random.rand(Sim['NSim'])#random numbers between 0 and 1 
    Sim['Wbar'][itime,:] = Sim['W'][itime,:]/Sim['Z'][itime]
    Sim['numberClones'] = np.maximum(np.floor(Sim['Wbar'][itime,:]+rnd),0)

def clone(Sim,itime,irep):
    # ~~~~ Get the difference between how many clones are created and how many total simulations should be there
    numberDiff = int(np.sum(Sim['numberClones']) - Sim['NSim'])

    # How many trajectories have numberClones>0
    Iavail = np.argwhere(Sim['numberClones']>0)
    numberAvail = len(Iavail) 

    # ~~~~ Balance the number of sim
    # If the number of sim is too high, remove some of them randomly
    if numberDiff>0:
        
        # Select simulations to kill
        toKill = ransam(numberAvail,numberDiff) 
        
        # Kill
        Sim['numberClones'][Iavail[toKill]] -= 1

    # ~~~~ Balance the number of sim
    #  If the number of sim is too low, add some of them randomly
    if numberDiff<0:
   
        # Select simulations to clone
        toClone = ransam_rep(numberAvail,-numberDiff)

        # Clone
        for indClone in list(toClone):
            Sim['numberClones'][Iavail[indClone]] += 1
    
    # ~~~~ Verify that the number of simulation is good
    if not np.sum(Sim['numberClones']) - Sim['NSim'] == 0:
        print("ERROR in clone: number of clones inconsistent with total number of Sim")
        sys.exit()


    #  ~~~~ Now, perform the cloning: assign the clone to the right simulations
    # Find the simulations that should be killed
    # These are the ones that will host the clones !

    # First get the number of simulations that are killed
    Ikilled = np.argwhere(Sim['numberClones']<=0)
    numberKilled = len(Ikilled) 

    # Get the simulations that are cloned
    Icloned = np.argwhere(Sim['numberClones']>1)

    # Monitor number of kills
    Sim['numberKills'][Sim['NselectionLoc']-1,irep] = numberKilled
   
    # Monitor distance between clone and ref
    Sim['diffClone'][Sim['NselectionLoc']-1,irep] =  0
    Sim['nClonesForDiff'][Sim['NselectionLoc']-1,irep] =  0
    

    # ~~~~ Now clone simulations 
    # Take a simulation to kill and replace it with a simulatiion to clone
    epsilonClone = Sim['Epsilon clone']
    if numberKilled >0 and np.amax(Sim['numberClones'])>1:
        counter = -1
        time_s = time.time()
        #fig = plt.figure()
        for iclone in list(Icloned):
            nclones = int(Sim['numberClones'][iclone] - 1)
            qoiVal = Sim['qoiFunc'](Sim['u'][:,iclone])
            if (not Sim['UseGAN']) or (itime<200) or (qoiVal>2.7) :
                for p in range(nclones): 
                    counter += 1
                    Sim['u'][:,Ikilled[counter]] = Sim['u'][:,iclone] + epsilonClone*np.random.normal(loc=0.0,
                                                                                                      scale=1.0,
                                                                                                      size=(Sim['u'].shape[0],1))
                    Sim['diffClone'][Sim['NselectionLoc']-1,irep] +=  np.mean(abs(Sim['u'][:,Ikilled[counter]] - Sim['u'][:,iclone]))
                    Sim['nClonesForDiff'][Sim['NselectionLoc']-1,irep] +=  1
                    Sim['qoiTot'][:itime,0,Ikilled[counter]] = Sim['qoiTot'][:itime,0,iclone]
                    Sim['qoiTot'][itime,0,Ikilled[counter]] = Sim['qoiFunc'](Sim['u'][:,Ikilled[counter]])
            else:
                #diff, clones = cloneSample(Sim['generator'],qoiVal,nclones,Sim['u'][:,iclone])
                #clones = closeCloneSample(Sim['generator'],qoiVal,nclones,Sim['u'][:,iclone])
                diff, clones = recursiveCloseCloneSample(Sim['generator'],qoiVal,nclones,Sim['u'][:,iclone])
                #print("clones shape",clones.shape)
                Sim['diffClone'][Sim['NselectionLoc']-1,irep] +=  diff*nclones
                Sim['nClonesForDiff'][Sim['NselectionLoc']-1,irep] +=  nclones
                for p in range(nclones):
                    counter += 1
                    Sim['u'][:,Ikilled[counter]] =  clones[p,:]
                    Sim['qoiTot'][:itime,0,Ikilled[counter]] = Sim['qoiTot'][:itime,0,iclone]
                    #print("p",p)
                    #print("clones[p,:] shape",clones[p,:].shape)
                    #print("Sim['qoiFunc'](clones[p,:])",Sim['qoiFunc'](clones[p,:]))
                    #print("clones[p,:]",clones[p,:])
                    Sim['qoiTot'][itime,0,Ikilled[counter]] = Sim['qoiFunc'](clones[p,:])
                    
                #plt.clf()
                #for p in range(nclones):
                #    plt.plot(clones[p,:],linewidth=1,color='b')
                #plt.plot(Sim['u'][:,iclone],linewidth=3,color='k')
                #plt.draw()
                #plt.pause(0.001)               

                Sim['numberClones'][iclone] -= 1
                Sim['numberClones'][Ikilled[counter]] += 1
        time_e = time.time()
        #print(time_e-time_s)

    # Verify that the number of simulation is good
    if not np.sum(Sim['numberClones']) == Sim['NSim']:
        print('ERROR in clone: number of clone inconsistent with NSim')
        sys.exit()
    


def prestep(u,Sim,itime): 
    if Sim['Ni'] == 0:
        Sim['timestepLastClone'] = itime-1#*Sim['Timestep']
        Sim['Elastclone'] = Sim['qoiFunc'](u)
    else:
        return 

def step(Sim): 
    Sim['Ni'] += 1

def poststep(Sim,itime,irep): 
    if not Sim['Ni'] == Sim['NselectionThreshold']:
        return
    computeWeights(Sim,itime)
    clone(Sim,itime,irep)


def finalize(Sim,irep):
    qoiEnd = Sim['qoiTot'][-1,0,:]
    qoiInit = Sim['qoiTot'][0,0,:]
    # Weights
    if not Sim['UseRarePath']:
        Sim['W'][-1,:] = np.exp(Sim['Cweight']*(qoiEnd-Sim['Elastclone']))
    else:
        C = computeRareFlucC(Sim,-1)
        ClastClone = computeRareFlucC(Sim,Sim['timestepLastClone'])
        Sim['W'][-1,:] = np.exp(C*qoiEnd-ClastClone*Sim['Elastclone'])
    
    #print("Finalize splitting")

    # Compute Z
    #Sim['Z'][-1] = 1
    Sim['Z'][-1] = np.mean(Sim['W'][-1,:])

    # Compute for each level
    for ilevel, level in enumerate(Sim['Levels'].tolist()):
        Sim['F_prob'] = np.zeros(Sim['NSim'])

        indLevel = np.argwhere(qoiEnd>=level)
        if not Sim['UseRarePath']:
            Sim['F_prob'][indLevel] = np.exp(Sim['Cweight']*(qoiInit[indLevel] - qoiEnd[indLevel]))
        else:
            CEnd = computeRareFlucC(Sim,-1)
            CInit = computeRareFlucC(Sim,0)
            Sim['F_prob'][indLevel] = np.exp(CInit*qoiInit[indLevel] - CEnd*qoiEnd[indLevel])

        productZ=1.0
        for itimestep in range(Sim['nmax']+1):
            if abs(Sim['Z'][itimestep])>1e-12:
                productZ = productZ*Sim['Z'][itimestep]

        sumF = np.sum(Sim['F_prob'])
       
        Sim['probabilities'][ilevel,irep] = sumF*productZ/Sim['NSim'] 
 
