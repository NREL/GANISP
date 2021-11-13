from plotsUtil import *
import parallel as par


def plotResult(Result,Sim):
    if Sim['Plot'] and par.irank==par.iroot:
        # Plot Result
        nmax = Sim['nmax']

        fig = plt.figure()
        plt.imshow(Result['uu'][:,:,0],aspect=Sim['Ndof']/(Sim['nmax']/Sim['nplt']),origin='lower',cmap='jet', interpolation='nearest')
        prettyLabels('x','t',14)
        plt.colorbar()

        fig=plt.figure()
        plt.plot(Result['tt'], Result['qoiTot'][:,0,:],linewidth=3,color='k')
        prettyLabels('T','Q',14)

        #np.savez('ldrd',tt=Result['tt'],qoiTot=Result['qoiTot'][:,0,:])
        

        if Sim['Simulation name'] in ['L96','KS']:
            fig = plt.figure()
            plt.plot(Result['uu'][-1,:,0],linewidth=3,color='k',label='end')
            prettyLabels('x','u',14)
            plotLegend()



        if Sim['Simulation name'] in ['L96FrontBack','KSFrontBack']:
            fig = plt.figure()
            plt.plot(Result['uu'][0,:],linewidth=3,color='k',label='start')
            plt.plot(Result['uu'][int(nmax/2),:],linewidth=3,color='b',label= str(int(nmax/2))+ r'$\Delta$ t')
            plt.plot(Result['uu'][-1,:],'--',linewidth=3,color='r',label=r'start recons')
            prettyLabels('x','u',14)
            plotLegend()


        plt.show()


def get1DCDF(x,val):
    
    if not len(val.shape)==1:
        print('in getCDF: val should be 1D') 
        sys.exit()
    if not len(x.shape)==1:
        print('in getCDF: s should be 1D') 
        sys.exit()
    
    n = len(x)
    N = len(val)
    y = np.zeros(n)
    
    previousVal = 0
    for i in range(n):
        y[i] = len(np.argwhere(val>=x[i]))/N
        
    return y

def buildCDF(Result,Sim):
      
 
    if Sim['Build CDF']:
        
        qoi_f = Result['qoiTot'][-1,0,:]
        if par.nProc>1 and not Sim['reconstruct QOI']:
            qoi_f = par.gather1DList(list(qoi_f),0,NSim)
 
        if par.irank==par.iroot:
            minval = np.amin(qoi_f)
            maxval = np.amax(qoi_f) + (np.amin(qoi_f) - np.amax(qoi_f))/1000
            
            xCDF = np.linspace(minval,maxval,100)
            yCDF = get1DCDF(xCDF,qoi_f)

            np.savez('CDF'+'_'+Sim['Simulation name']+'_'+str(Sim['NSim']),xCDF=xCDF,yCDF=yCDF)

            if Sim['Plot CDF']:
                # Plot
                fig = plt.figure()
                plt.plot(xCDF,yCDF,linewidth=3,color='k')
                ax = plt.gca()
                ax.set_yscale('log')
                prettyLabels('a','P(Q>a)',14)
                plt.show()



def buildRarePaths(Result,Sim):
      
 
    if Sim['Build rare paths']:
       
        rareLevels = Sim['Levels']
        nmax = Sim['nmax']
        NSim = Sim['NSim']   
        nSim_ = Sim['nSim_']   
 
        rarePaths = np.zeros((nmax+1,len(rareLevels))) 
        nPaths = np.zeros(len(rareLevels))
        meanPath = np.zeros(nmax+1) 
        meanPath2 = np.zeros(nmax+1) 
        stdPath = np.zeros(nmax+1) 
 
        qoi_f = Result['qoiTot'][-1,0,:]

        if par.nProc==1 or Sim['reconstruct QOI']:
            meanPath = np.mean(Result['qoiTot'][:,0,:],axis=1)[:]
            stdPath = np.std(Result['qoiTot'][:,0,:],axis=1)[:]
            for ilev, lev in enumerate(rareLevels):
                ind = np.argwhere(qoi_f>lev)
                if len(ind)>1:
                    rarePaths[:,ilev] = np.mean(Result['qoiTot'][:,0,ind],axis=1)[:,0]
                    nPaths[ilev] = len(ind)
                elif len(ind)==1:
                    rarePaths[:,ilev] = np.squeeze(Result['qoiTot'][:,0,ind])
                    nPaths[ilev] = len(ind)
                else:
                    rarePaths[:,ilev] = np.zeros(nmax+1)
                    nPaths[ilev] = 0
            
        else:
            # Did not reconstruct QOI
            meanPath_ = np.mean(Result['qoiTot'][:,0,:],axis=1)[:]
            meanPath2_ = np.mean(Result['qoiTot'][:,0,:]**2,axis=1)[:]
            meanPathGlob = par.gatherMulti1DList(meanPath_,0,nmax+1)
            meanPath2Glob = par.gatherMulti1DList(meanPath2_,0,nmax+1)
            nSimGlob = np.array(par.comm.gather(nSim_,root=0),dtype=float)              
            if par.irank==par.iroot:
                meanPath = np.sum(np.moveaxis(meanPathGlob,0,-1)*nSimGlob,axis=1)/float(NSim)
                meanPath2 = np.sum(np.moveaxis(meanPath2Glob,0,-1)*nSimGlob,axis=1)/float(NSim)
                stdPath = np.sqrt(meanPath2 - meanPath**2)

            for ilev, lev in enumerate(rareLevels):
                
                ind = np.argwhere(qoi_f>lev)
                if len(ind)>1:
                    rarePath_ = np.mean(Result['qoiTot'][:,0,ind],axis=1)[:,0]
                    nPath_ = len(ind)
                elif len(ind)==1:
                    rarePath_ = Result['qoiTot'][:,0,ind]
                    nPath_ = len(ind)
                else:
                    rarePath_ = np.zeros(nmax+1)
                    nPath_ = 0

                rarePathGlob = par.gatherMulti1DList(list(rarePath_),0,nmax+1)
                nPathGlob =    np.array(par.comm.gather(nPath_,root=0))

                if par.irank==par.iroot:
                    for iproc in range(par.nProc):
                        rarePaths[:,ilev] += np.array(rarePathGlob[iproc])*int(nPathGlob[iproc])
                        nPaths[ilev] += nPathGlob[iproc]

                    if nPaths[ilev]>0:
                        rarePaths[:,ilev] /= nPaths[ilev]
                    else: 
                        rarePaths[:,ilev] = 0

        if par.irank==par.iroot:
            for ilev, lev in enumerate(rareLevels):
                np.savez('Path'+'_'+Sim['Simulation name']+'_'+str(lev),rarePath=rarePaths[:,ilev],nPaths=nPaths[ilev],meanPath=meanPath,stdPath=stdPath,NSim=NSim,tt=Result['tt'])

            if Sim['Plot rare paths']:
                # Plot
                fig = plt.figure()
                for ilev, lev in enumerate(rareLevels):
                    plt.plot(Result['tt'],rarePaths[:,ilev],linewidth=3,color='k')
                plt.plot(Result['tt'],meanPath,linewidth=3,color='r')
                plt.plot(Result['tt'],meanPath+stdPath,'--',linewidth=3,color='r')
                plt.plot(Result['tt'],meanPath-stdPath,'--',linewidth=3,color='r')
                prettyLabels('tt','Q',14)
                plt.show()


def postProc(Result,Sim):
    plotResult(Result,Sim)
    buildCDF(Result,Sim)
    buildRarePaths(Result,Sim)

