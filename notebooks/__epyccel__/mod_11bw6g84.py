@types('float[:,:]', 'float[:,:]','int','float','float','float','float')
def solve_2d_diff_pyccel(u, un, nt, dt, dx, dy, nu):
  row, col = u.shape
    
  u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
  for j in range(0, nt + 1):
     for ix in range(1, x_points-1):
        for iy in range(1, y_points-1):
            u_new[ix, iy] = u[ix, iy] + nu* (dt/ dx**2 )* (
                        u[ix + 1,iy] - 2 * u[ix,iy] + u[ix - 1,iy]) + (nu * dt / dy**2) * (
                                        u[ix,iy + 1] - 2 * u[ix,iy] + u[ix,iy - 1])

            u_new[0,:] = 0
            u_new[:, 0] = 0
            u_new[-1, :] = 0
            u_new[:, -1] = 0

    
    #fill the update of u and v
        
  return 0
