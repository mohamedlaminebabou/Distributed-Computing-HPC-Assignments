#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:10:51 2021

@author: babou
"""
from mpi4py import MPI
from math import pi ,pow

def s(n):
  sn=0.
  for k in range(n):
        sn+=4*((-1)**k)/(2*k +1) 
  return sn
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
nproc= COMM.Get_size()


if(RANK == 0):
   n=10**8
else:
   n=None
n=COMM.bcast(n, root=0)
pi_app=s(n)/2
R=COMM.reduce(pi_app , op=MPI.SUM, root=0)
if RANK==0:
    print ('approximation for pi=  %.25f' % (R))
    print('exact pi=  %.25f' % (pi))
    print('error= %.25f'% (abs(pi-R)))
 
