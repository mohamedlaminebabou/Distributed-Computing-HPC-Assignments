# Monte carlo to calculate pi



from numpy import random 
from mpi4py import MPI
from math import pi

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

print (RANK, 'uses seed', RANK)

N = 10**9# N = 10**7 , N = 10**8
k = 0
for i in range(0, N):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    if x**2 + y**2 <= 1:
        k = k + 1
R = float(k)/N

print (RANK, 'computes', R)

if(RANK == 1):
    COMM.send(R, dest=0, tag=11)
    print (RANK, 'sends', R, 'to 0')
elif(RANK == 0):
    S = COMM.recv(source=1, tag=11)
    print (RANK, 'received', S, 'from 1')
    RESULT = 2*(R + S)
    print ('approximation for pi =   %.12f' % RESULT)
    print('exact pi= %.12f'% pi)
    print('error= %.12f'%(pi-RESULT))
