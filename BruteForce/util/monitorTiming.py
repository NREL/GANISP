import numpy as np
import parallel as par

def printTiming(Result):
    elapsedTimeTot = par.allmaxScalar(Result['timeExec'])
    elapsedTimeInit = par.allmaxScalar(Result['timeExecInit'])
    elapsedTimeMain = par.allmaxScalar(Result['timeExecMain'])
    elapsedTimeStep = par.allmaxScalar(Result['timeExecStep'])
    elapsedTimeRecons = par.allmaxScalar(Result['timeExecRecons'])

    par.printRoot("Exec in: " + " %.3f s" % elapsedTimeTot)
    par.printRoot("\t init: " + " %.3f" % (elapsedTimeInit*100/elapsedTimeTot) + "%")
    par.printRoot("\t main: " + " %.3f" % (elapsedTimeMain*100/elapsedTimeTot) + "%")
    par.printRoot("\t step: " + " %.3f" % (elapsedTimeStep*100/elapsedTimeTot) + "%")
    par.printRoot("\t recons: " + " %.3f" % (elapsedTimeRecons*100/elapsedTimeTot) + "%")
