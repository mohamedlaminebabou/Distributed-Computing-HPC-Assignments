@types('float[:,:](order=C)','float[:,:](order=F)','float[:,:](order=C)')
def dot(a, b, c):
    m, p = a.shape
    q, n = b.shape
    r, s = c.shape
    if p != q or m != r or n != s:
        return-1
    #$ omp parallel
    #$ omp do schedule(runtime)
    for i in range(m):
        for j in range(n):
            c[i, j] = 0.0
            for k in range(p):
                c[i, j] += a[i, k] * b[k, j]
    #$ omp end do
    #$ omp end parallel
    return 0
