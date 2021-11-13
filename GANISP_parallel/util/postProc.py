from plotsUtil import *
import os 
import parallel as par

def plotResult(Result,Sim):
    
    if not par.irank==par.iroot:
        return
 
    if Sim['Plot']:
        # Plot Result
        nmax = Sim['nmax']

        fig = plt.figure()
        plt.imshow(Result['uu'][:,:,0],aspect=Sim['Ndof']/(Sim['nmax']/Sim['nplt']),origin='lower',cmap='jet', interpolation='nearest')
        prettyLabels('x','t',14)
        plt.colorbar()

        fig=plt.figure()
        plt.plot(Result['tt'], Result['qoiTot'][:,0,:],linewidth=0.5,color='k')
        prettyLabels('T','Q',14)


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


        #plt.show()

def plotKillHistory(Sim):

    sumNumberKills = np.sum(Sim['numberKills'],axis=1)
    allsumNumberKills = par.allsum1DArrays(sumNumberKills)
    allmeanNumberKills = allsumNumberKills/Sim['NRep']
    
    if not par.irank==par.iroot:
        return

    if Sim['Plot kill history']:
        fig = plt.figure()
        plt.plot(allmeanNumberKills/Sim['NSim'],'-o',color='b',linewidth=3,markersize=10)
        prettyLabels('Selection Step','# kills/ # Sim',14)
        #plt.show()

def plotCloneHistory(Sim):

    sumCloneDiff = np.sum(Sim['diffClone'],axis=1)
    meanCloneDiff = par.allsum1DArrays(sumCloneDiff)
    sumNClone = np.sum(Sim['nClonesForDiff'],axis=1)
    meanNClone = par.allsum1DArrays(sumNClone)
    
    if not par.irank==par.iroot: 
        return

    if Sim['Plot clone history']:
        for i in range(len(meanNClone)):
            if meanNClone[i]>0:
                meanCloneDiff[i] /=  meanNClone[i]
            else:
                meanCloneDiff[i] =  0
                  
        fig = plt.figure()
        plt.plot(meanCloneDiff,'-o',color='b',linewidth=3,markersize=10)
        prettyLabels('Selection Step','Average Diff Clone',14)
        #plt.show()

def plotISPCDF(Sim):
 
    sumISP = np.sum(Sim['probabilities'],axis=1)
    meanISP = par.allsum1DArrays(sumISP)/Sim['NRep']
    ISPtoMeanSQ = (Sim['probabilities'] - np.reshape(meanISP,(Sim['Number of thresholds'],1)))**2
    stdISP = np.sqrt(par.allsum1DArrays(np.sum(ISPtoMeanSQ,axis=1))/Sim['NRep'])


    sumNumberKills = np.sum(Sim['numberKills'],axis=1)
    allsumNumberKills = par.allsum1DArrays(sumNumberKills)
    allmeanNumberKills = allsumNumberKills/(Sim['NRep']*Sim['NSim'])


    sumCloneDiff = np.sum(Sim['diffClone'],axis=1)
    meanCloneDiff = par.allsum1DArrays(sumCloneDiff)
    sumNClone = np.sum(Sim['nClonesForDiff'],axis=1)
    meanNClone = par.allsum1DArrays(sumNClone)
    for i in range(len(meanNClone)):
        if meanNClone[i]>0:
            meanCloneDiff[i] /=  meanNClone[i]
        else:
            meanCloneDiff[i] =  0

    if not par.irank==par.iroot:
        return
 
    if Sim['Plot ISP CDF']:

        trueCDF = np.load(Sim['True CDF file'])
        trueX = trueCDF['xCDF']
        trueY = trueCDF['yCDF']
        trueYInterp = np.interp(Sim['Levels'],trueX,trueY)

        meanBrute = trueY
        stdBrute = np.sqrt((trueY -trueY**2)/Sim['NSim'])

        meanBruteInterp = trueYInterp
        stdBruteInterp = np.sqrt((trueYInterp -trueYInterp**2)/Sim['NSim'])
 

        fig = plt.figure()
        plt.plot(Sim['Levels'], meanISP,color='b',linewidth=3,label='ISP')
        plt.plot(Sim['Levels'], meanISP + stdISP,'--',color='b',linewidth=3)
        plt.plot(Sim['Levels'], meanISP - stdISP,'--',color='b',linewidth=3)
        plt.plot(trueX,         meanBrute,color='k',linewidth=3,label='Truth')
        plt.plot(trueX,         meanBrute + stdBrute,'--',color='k',linewidth=3)
        plt.plot(trueX,         meanBrute - stdBrute,'--',color='k',linewidth=3)
        ax = plt.gca()
        plt.ylim([4e-7,1])
        ax.set_yscale('log')
        plotLegend()
        prettyLabels('Q(Tf)','P',14)
        
        plt.savefig('Figures/Est'+Sim['Simulation name']+'_C_'+str(Sim['Cweight'])+
                    '_N_'+str(Sim['Nselection'])+
                    '_eps_'+str(Sim['Epsilon clone'])+
                    '.png')


        fig = plt.figure()
        plt.plot(trueYInterp,stdBruteInterp/stdISP,color='k',linewidth=3,label='Computational Gain')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plotLegend()
        prettyLabels('P','Std Brute Force / Std ISP',14)
        plt.savefig('Figures/Gain'+Sim['Simulation name']+'_C_'+str(Sim['Cweight'])+
                    '_N_'+str(Sim['Nselection'])+
                    '_eps_'+str(Sim['Epsilon clone'])+
                    '.png')


        fig = plt.figure()
        plt.plot(trueYInterp,abs(meanISP-meanBruteInterp)/meanBruteInterp,color='k',linewidth=3,label='Computational Gain')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plotLegend()
        prettyLabels('P','Std Brute Force / Std ISP',14)
        plt.savefig('Figures/Biais'+Sim['Simulation name']+'_C_'+str(Sim['Cweight'])+
                    '_N_'+str(Sim['Nselection'])+
                    '_eps_'+str(Sim['Epsilon clone'])+
                    '.png')


 

        np.savez(Sim['Simulation name']+'_C_'+str(Sim['Cweight'])+
                 '_N_'+str(Sim['Nselection'])+
                 '_eps_'+str(Sim['Epsilon clone'])+
                 '.npz',
                 bruteProb=trueYInterp,
                 bruteProbStd=stdBruteInterp,
                 ispProb=meanISP,
                 ispProbStd=stdISP,
                 relBiais=abs(meanISP-meanBruteInterp)/meanBruteInterp,
                 compGain=stdBruteInterp/stdISP,
                 meanCloneDiff=meanCloneDiff,
                 allmeanNumberKills=allmeanNumberKills)

        plt.show()

def reproducePlots(Sim):
    if Sim['Plot ISP CDF']:
        trueCDF = np.load(Sim['True CDF file'])
        trueX = trueCDF['xCDF']
        trueY = trueCDF['yCDF']
        trueYInterp = np.interp(Sim['Levels'],trueX,trueY)


        meanISP = np.mean(Sim['probabilities'],axis=1)
        stdISP = np.std(Sim['probabilities'],axis=1)
      
        meanBrute = trueY
        stdBrute = np.sqrt((trueY -trueY**2)/Sim['NSim'])

        meanBruteInterp = trueYInterp
        stdBruteInterp = np.sqrt((trueYInterp -trueYInterp**2)/Sim['NSim'])
 

        fig = plt.figure()
        plt.plot(Sim['Levels'], meanISP,color='b',linewidth=3,label='ISP')
        plt.plot(Sim['Levels'], meanISP + stdISP,'--',color='b',linewidth=3)
        plt.plot(Sim['Levels'], meanISP - stdISP,'--',color='b',linewidth=3)
        plt.plot(trueX,         meanBrute,color='k',linewidth=3,label='Truth')
        plt.plot(trueX,         meanBrute + stdBrute,'--',color='k',linewidth=3)
        plt.plot(trueX,         meanBrute - stdBrute,'--',color='k',linewidth=3)
        ax = plt.gca()
        ax.set_yscale('log')
        plotLegend()
        prettyLabels('Q(Tf)','P',14)

        fig = plt.figure()
        plt.plot(Sim['Levels'], meanISP,color='b',linewidth=3,label='ISP')
        plt.plot(Sim['Levels'], meanISP + stdISP,'--',color='b',linewidth=3)
        plt.plot(Sim['Levels'], meanISP - stdISP,'--',color='b',linewidth=3)
        plt.plot(trueX,         meanBrute,color='k',linewidth=3,label='Truth')
        plt.plot(trueX,         meanBrute + stdBrute,'--',color='k',linewidth=3)
        plt.plot(trueX,         meanBrute - stdBrute,'--',color='k',linewidth=3)
        ax = plt.gca()
        ax.set_xlim([1400, 1900])
        ax.set_ylim([1e-6, 0.02])
        ax.set_yscale('log')
        plotLegend()
        prettyLabels('Q(Tf)','P',14)


        fig = plt.figure()
   
        plt.plot(Sim['Levels'], stdISP/meanBruteInterp,color='b',linewidth=3, label='ISP')
        plt.plot(Sim['Levels'], stdBruteInterp/meanBruteInterp,color='k',linewidth=3, label='Brute')
        ax = plt.gca()
        ax.set_xlim([1000, 1900])
        ax.set_ylim([0,5])
        plotLegend()
        prettyLabels('a','RE',14)
        plt.show()
 
        #np.savez('data/Nomean_C'+str(Sim['Cweight'])+'.npz',bruteRe=stdBruteInterp/meanBruteInterp,ispRe=stdISP/meanBruteInterp,levels=Sim['Levels'])
        #np.savez('data/Mean_C'+str(Sim['Cweight'])+'.npz',bruteRe=stdBruteInterp/meanBruteInterp,ispRe=stdISP/meanBruteInterp,levels=Sim['Levels'])




def postProc(Result,Sim):
    os.makedirs('Figures',exist_ok=True)
    plotResult(Result,Sim)
    plotKillHistory(Sim)
    plotCloneHistory(Sim)
    plotISPCDF(Sim)
    #reproducePlots(Sim)
