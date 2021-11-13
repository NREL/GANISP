import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

def prettyLabels(xlabel,ylabel,fontsize,title=None):
    plt.xlabel(xlabel, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    plt.ylabel(ylabel, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    if not title==None:
        plt.title(title, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()

def plotLegend(fontsize=16):
    fontsize = fontsize
    plt.legend()
    leg=plt.legend(prop={'family':'Times New Roman','size': fontsize-3,'weight':'bold' })
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor('k')    

def snapVizZslice(field,x,y,figureDir, figureName,title=None):
    fig,ax = plt.subplots(1)
    plt.imshow(np.transpose(field), cmap=cm.jet, interpolation='bicubic', vmin=np.amin(field), vmax=np.amax(field), extent=[np.amin(x),np.amax(x),np.amax(y),np.amin(y)])
    prettyLabels("x [m]","y [m]", 16, title) 
    plt.colorbar()
    fig.savefig(figureDir+'/'+figureName)
    plt.close(fig)
    return 0

def movieVizZslice(field,x,y,itime,movieDir,minVal=None,maxVal=None):
    fig,ax = plt.subplots(1)
    fontsize =  16
    if minVal==None:
        minVal=np.amin(field)
    if maxVal==None:
        maxVal=np.amax(field)
    plt.imshow(np.transpose(field), cmap=cm.jet, interpolation='bicubic', vmin=minVal, vmax=maxVal, extent=[np.amin(x),np.amax(x),np.amax(y),np.amin(y)])
    prettyLabels("x [m]","y [m]", 16, 'Snap Id = ' + str(itime))
    plt.colorbar()
    fig.savefig(movieDir+'/im_'+str(itime)+'.png')
    plt.close(fig)
    return 0

def makeMovie(ntime,movieDir,movieName):
    fig = plt.figure()
    # initiate an empty  list of "plotted" images 
    myimages = []
    #loops through available png:s
    for i in range(ntime):
        ## Read in picture
        fname = movieDir+"/im_"+str(i)+".png"
        myimages.append(imageio.imread(fname))
    imageio.mimsave(movieName, myimages)
    return


def plotHist(field,xLabel,folder, filename):
    fig=plt.figure()
    plt.hist(field)
    fontsize = 18
    prettyLabels(xLabel,"bin count", fontsize)
    fig.savefig(folder + '/' + filename)

def plotContour(x,y,z,color):
    ax =plt.gca()
    X,Y = np.meshgrid(x,y)
    CS = ax.contour(X, Y, np.transpose(z), [0.001, 0.005, 0.01 , 0.05], colors=color)
    h,_ = CS.legend_elements()
    return h[0]

def plotActiveSubspace(paramName,W,title=None):
    x=[]
    for i,name in enumerate(paramName):
        x.append(i)
    fig = plt.figure()
    plt.bar(x, W, width=0.8, bottom=None, align='center', data=None, tick_label=paramName)
    fontsize = 16
    if not title==None:
        plt.title(title, fontsize=fontsize, fontweight='bold', fontname="Times New Roman")
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
        #ax.spines[axis].set_zorder(0)
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
