@types('float[:,:]', 'float[:,:]','int','float','float','float','float')

def solve_2d_linearconv_pyccel(u, un, nt, dt, dx, dy, c):
   
   for n in range(nt + 1): ##loop across number of time steps
    un[:,:] = u[:,:]
    row, col = u.shape
    for j in range(1, row):
        for i in range(1, col):
            u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                  (c * dt / dy * (un[j, i] - un[j - 1, i])))
            u[0, :] = 1
            u[-1, :] = 1
            u[:, 0] = 1
            u[:, -1] = 1    

    
               
    return 0
