@types('float[:]', 'float[:]','int','int','float','float','float')

def solve_1d_burger_pyccel(u, un, nt, nx, dt, dx, nu):
    for n in range(nt):
     un[:] = u[:]
     for i in range(1, nx-1):
        u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 *\
                 (un[i+1] - 2 * un[i] + un[i-1])
     u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 *\
                (un[1] - 2 * un[0] + un[-2])
     u[-1] = u[0]
        
    return 0
