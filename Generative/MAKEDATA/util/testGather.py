import numpy as np
import parallel as par


n = par.irank+1

n = par.comm.gather(n, root=0)


par.printAll("n  ",n)


A = np.array([1,2])*par.irank

A = par.gatherMulti1DList(list(A), 0, 2)
par.printAll("A  ",A)

