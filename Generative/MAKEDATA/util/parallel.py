from mpi4py import MPI
import numpy as np
import sys


# MPI Init
# MPI.Init() called when MPI is imported
comm = MPI.COMM_WORLD
irank = comm.Get_rank()+1
iroot = 1
nProc = comm.Get_size()
status = MPI.Status()
   



# ~~~~ Print functions
def printRoot(description, item=None):
    if irank == iroot:
        if type(item).__name__ == 'NoneType':
            print(description)
        elif (not type(item).__name__ == 'ndarray'):
            print(description + ': ',item)
        else:
            print(description + ': ',item.tolist())
        sys.stdout.flush()
    return

def printAll(description, item=None):
    if type(item).__name__ == 'NoneType':
        print('[' + str(irank) + '] ' + description)
    elif (not type(item).__name__ == 'ndarray'):
        print('[' + str(irank) + '] ' + description + ': ',item)
    else:
        print('[' + str(irank) + '] ' + description + ': ', item.tolist())
    sys.stdout.flush()
    return

# ~~~~ Partition functions
def partitionSim(nSim):
    # ~~~~ Partition the files with MPI
    # Simple parallelization across simulations
    NSimGlob = nSim
    tmp1=0
    tmp2=0
    for iproc in range(nProc):
        tmp2 = tmp2 + tmp1
        tmp1 = int(NSimGlob/(nProc-iproc))
        if irank == (iproc+1):
            nSim_ = tmp1
            startSim_ = tmp2
        NSimGlob = NSimGlob - tmp1
    return nSim_, startSim_

# ~~~~ Reduce functions
def gather1DList(list_,rootId,N):
    list_=np.array(list_,dtype='double')
    sendbuf = list_
    recvbuf = np.empty(N,dtype='double')
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    comm.Gatherv(sendbuf, recvbuf=(recvbuf, sendcounts), root=rootId)
    return recvbuf

def gather2DList(list_,rootId,N1Loc, N1Glob, N2):
    # ~~~ The parallelization is across the axis 0
    # ~~~ This will not work if the parallelization is across axis 1
    list_=np.array(list_,dtype='double')
    # Reshape the local data matrices:
    nElements_ = N1Loc * N2
    sendbuf = np.reshape(list_, nElements_, order='C')
    # Collect local array sizes:
    sendcounts = comm.gather(len(sendbuf), root=rootId)
    # Gather the data matrix:
    recvbuf = np.empty(N1Glob*N2, dtype='double')
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
    recvbuf = np.reshape(recvbuf, (N1Glob, N2), order='C')
    return recvbuf

def allgather1DList(list_,N):
    list_=np.array(list_,dtype='double')
    # Collect local array sizes:
    recvbuf = np.empty(N,dtype='double')
    comm.Allgatherv(list_, recvbuf)
    return recvbuf

def gatherMulti1DList(list_,rootId,N):
    error = False
    if not len(list_)==N:
       print("["+str(par.irank)+"]"+ " in gatherMulti1DList: len(list)="+ str(len(list_))+ " but expected "+ str(N))
       sys.stdout.flush()
       error = True
    if error:
       printAll("All proc should pass the same length of list")
       sys.exit()
    return np.array(comm.gather(list_, root=rootId),dtype=float)

def allsum1DArrays(A):
    buf = np.zeros(len(A),dtype='double') 
    comm.Allreduce(A, buf, op=MPI.SUM)
    return buf

def allsumMultiDArrays(A):
    # Takes a 3D array as input
    # Returns 3D array
    shapeDim = A.shape
    nTotDim = int(np.prod(shapeDim))
    buf = np.zeros(nTotDim,dtype='double') 
    comm.Allreduce(np.reshape(A,nTotDim), buf, op=MPI.SUM)
    return np.reshape(buf,shapeDim)


def allsumScalar(A):
    result = comm.allreduce(A, op=MPI.SUM)
    return result

def allmaxScalar(A):
    result = comm.allreduce(A, op=MPI.MAX)
    return result

def bcast(A):
    A = comm.bcast(A, root=0)
    return A

def reconstruct(uu,qoiTot,Sim):
    uuGlob =uu
    qoiTotGlob = qoiTot
    nSim_ = Sim['nSim_']
    NSim = Sim['NSim']
    nmax = Sim['nmax']
    if nProc>1 and Sim['reconstruct QOI']:
       qoiTot_ = np.reshape(qoiTot, (nmax+1,nSim_))
       qoiTot_ = np.moveaxis(qoiTot, -1, 0)
       qoiTotGlob = gather2DList(list(qoiTot_),0,nSim_, NSim, nmax+1)
       qoiTotGlob = np.moveaxis(qoiTotGlob,0,-1)
       qoiTotGlob = np.reshape(qoiTotGlob,(nmax+1,1,NSim))

    return uuGlob, qoiTotGlob


def finalize():
    MPI.Finalize()
