@types('float[:,:]', 'float[:,:]','int','float','float','float','float')
def solve_2d_diff_pyccel(u, un, nt, dt, dx, dy, nu):
  row, col = u.shape
    
  u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
 
  for j in range(0, nt + 1):
     un[:,:]=u[:,:]
     for ix in range(1, row):
        for iy in range(1, col):
            un[ix, iy] = u[ix, iy] + nu* (dt/ dx**2 )* (
                        u[ix + 1,iy] - 2 * u[ix,iy] + u[ix - 1,iy]) + (nu * dt / dy**2) * (
                                        u[ix,iy + 1] - 2 * u[ix,iy] + u[ix,iy - 1])

     u[0, :] = 1
     u[-1, :] = 1
     u[:, 0] = 1
     u[:, -1] = 1
    
    #fill the update of u and v
        
  return 0
