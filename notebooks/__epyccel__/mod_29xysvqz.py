@types('float[:,:]', 'float[:,:]','float[:,:]','float[:,:]', 'int' ,'float', 'float','float', 'float')

def solve_2d_burger_pyccel(u, un, v, vn, nt, dt, dx, dy, nu):
       ###Assign initial conditions
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
 v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
 row, col = u.shape
 un [:,:]= u[:,:]
 vn[:,:] = v[:,:]

          
 for nt in range(0, nt + 1):
    for ix in range(1, row-1):
        for iy in range(1, col-1):
            u[ix, iy] = un[ix, iy] + (nu) * dt/(dx**2) * (
                    un[ix + 1, iy] - 2 * un[ix, iy] + un[ix - 1, iy]) + (nu) * dt/(dy**2) * (
                                        un[ix, iy + 1] - 2 * un[ix, iy] + un[ix, iy - 1]) - un[ix, iy] * (
                                        dt / float(dx)) * (un[ix, iy] - un[ix - 1, iy]) - vn[ix, iy] * (
                                        dt / float(dx)) * (un[ix, iy] - un[ix, iy - 1])   

            v[ix, iy] = vn[ix, iy] + nu* (dt / dx**2) * (
                    vn[ix + 1, iy] - 2 * vn[ix, iy] + vn[ix - 1, iy]) + (nu) * (dt/dy**2) * (
                                    vn[ix, iy + 1] - 2 * vn[ix, iy] + vn[ix, iy - 1]) - un[ix, iy] * (
                                    dt/ float(dx)) * (vn[ix, iy] - vn[ix - 1, iy]) - vn[ix, iy] * (
                                    dt/ float(dy)) * (vn[ix, iy] - vn[ix, iy - 1])

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1 
    
    
  
        
        
 return 0
