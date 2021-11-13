from plotsUtil import *
import os
import parallel as par

def plotResult(Result,Sim):

    if not par.irank == par.iroot:
        return

    if Sim['Plot']:
        # Plot Result
        nmax = Sim['nmax']

        fig = plt.figure()
        plt.imshow(Result['uu'][:,:,0],aspect=Sim['Ndof']/(Sim['nmax']/Sim['nplt']),origin='lower',cmap='jet', interpolation='nearest')
        if Sim['Simulation name']=='L96':
            #prettyLabels(r'$\xi_i$','t',30)
            prettyLabels('','t',30)
        if Sim['Simulation name']=='KS':
            prettyLabels('x','t',30)
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        #plt.colorbar()
        cbar=plt.colorbar()
        cbar.set_label(r'$\xi_i$')
        ax = cbar.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family='times new roman', weight='bold', size=25)
        text.set_font_properties(font)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(20)
        fig.tight_layout()
        plt.savefig('Figures/XTContour'+Sim['Simulation name']+'_C_'+str(Sim['Cweight'])+
                    '_N_'+str(Sim['Nselection'])+
                    '_eps_'+str(Sim['Epsilon clone'])+
                    '.png')
        plt.savefig('Figures/XTContour'+Sim['Simulation name']+'_C_'+str(Sim['Cweight'])+
                    '_N_'+str(Sim['Nselection'])+
                    '_eps_'+str(Sim['Epsilon clone'])+
                    '.eps')
 
        fig=plt.figure()
        plt.plot(Result['tt'], Result['qoiTot'][:,0,:],linewidth=3,color='k')
        prettyLabels('t','Q',14)


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
 
def plotISPCDF(Sim):

    sumISP = np.sum(Sim['probabilities'],axis=1)
    meanISP = par.allsum1DArrays(sumISP)/Sim['NRep']
    ISPtoMeanSQ = (Sim['probabilities'] - np.reshape(meanISP,(Sim['Number of thresholds'],1)))**2
    stdISP = np.sqrt(par.allsum1DArrays(np.sum(ISPtoMeanSQ,axis=1))/Sim['NRep'])

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
        plt.plot(trueYInterp,abs(meanISP-meanBruteInterp)/meanBruteInterp,color='k',linewidth=3,label='Bias')
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
                 compGain=stdBruteInterp/stdISP)

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
 
        #np.savez('data/Nomean_C'+str(Sim['Cweight'])+'.npz',bruteRe=stdBruteInterp/meanBruteInterp,ispRe=stdISP/meanBruteInterp,levels=Sim['Levels'])
        #np.savez('data/Mean_C'+str(Sim['Cweight'])+'.npz',bruteRe=stdBruteInterp/meanBruteInterp,ispRe=stdISP/meanBruteInterp,levels=Sim['Levels'])




def postProc(Result,Sim):
    os.makedirs('Figures',exist_ok=True)
    plotResult(Result,Sim)
    plotKillHistory(Sim)
    plotISPCDF(Sim)
    #reproducePlots(Sim)
    plt.show()
