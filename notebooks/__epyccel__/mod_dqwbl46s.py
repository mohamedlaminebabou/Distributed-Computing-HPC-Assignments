def solve_2d_nonlinearconv_pyccel(u, un, v, vn, nt, dt, dx, dy, c):

    ###Assign initial conditions
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    ##set hat function I.C. : v(.5<=x<=1 && .5<=y<=1 ) is 2
    v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    row, col = u.shape
    
    for it in range(0, nt):
           un [:,:]= u[:,:]
           vn[:,:] = v[:,:]
           for i in range(1, row):
                for j in range(1, col): 
    
                  u[i,j]=u[i,j]-u[i,j]*(dt/dx)*\
                          (u[i,j]-u[i-1,j])-v[i,j]*(dt/dy)*(u[i,j]-u[i,j-1])
                  v[i,j]=v[i,j]-u[i,j]*(dt/dx)* \
                          (v[i,j]-v[i-1,j])-v[i,j]*(dt/dy)*(v[i,j]-v[i,j-1])
           u[0, :] = 1
           u[-1, :] = 1
           u[:, 0] = 1
           u[:, -1] = 1
    
           v[0, :] = 1
           v[-1, :] = 1
           v[:, 0] = 1
           v[:, -1] = 1

        
    return 0
