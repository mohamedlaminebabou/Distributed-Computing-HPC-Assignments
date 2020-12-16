@types('float[:]', 'float[:]','int','int','float','float','float')

def solve_1d_nonlinearconv_pyccel(u, un, nt, nx, dt, dx):
    L=dt/dx
    for j in range(nt):
        un=[u[i] for i in range(nx)]
        for i in range(1,nx): 
           un[i]=un[i] + L*un[i]*(un[i]-un[i-1])  
   
    return 0
