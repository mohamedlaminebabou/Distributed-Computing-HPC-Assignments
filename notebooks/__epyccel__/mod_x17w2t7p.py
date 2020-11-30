@types('float[:]', 'float[:]','int','int','float','float','float')
def solve_1d_linearconv_f90(u, un, nt, nx, dt, dx, c):
    for j in range(nt):
        un=u
        for i in range(nx-1): 
            u[i+1]=(1-c*(dt/dx))*un[i+1] +c*(dt/dx)*un[i]   # or un[i]=(1-L)*un[i] +L*un[i-1]  "and for start from 1 "
    return 0
