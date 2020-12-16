module mod_lh1i7w9c

use ISO_C_BINDING

implicit none

contains

!........................................
function solve_1d_nonlinearconv_pyccel(u, un, nt, nx, dt, dx) result( &
      Out_0001)

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(in) :: u(0:)
  real(C_DOUBLE), intent(inout) :: un(0:)
  integer(C_LONG_LONG), value :: nt
  integer(C_LONG_LONG), value :: nx
  real(C_DOUBLE), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE) :: L
  integer(C_LONG_LONG) :: j
  integer(C_LONG_LONG) :: i
  integer(C_LONG_LONG) :: Dummy_0001

  L = dt / dx
  do j = 0_C_LONG_LONG, nt-1_C_LONG_LONG, 1_C_LONG_LONG
    if (allocated(un)) then
      if (any(size(un) /= [nx])) then
        deallocate(un)
        allocate(un(0:nx - 1_C_LONG_LONG))
      end if
    else
      allocate(un(0:nx - 1_C_LONG_LONG))
    end if
    Dummy_0001 = 0_C_LONG_LONG
    do i = 0_C_LONG_LONG, nx-1_C_LONG_LONG, 1_C_LONG_LONG
      un(Dummy_0001) = u(i)
      Dummy_0001 = Dummy_0001 + 1_C_LONG_LONG
    end do
    do i = 1_C_LONG_LONG, nx-1_C_LONG_LONG, 1_C_LONG_LONG
      un(i) = un(i) + L * un(i) * (un(i) - un(i - 1_C_LONG_LONG))
    end do
  end do
  Out_0001 = 0_C_LONG_LONG
  return

end function solve_1d_nonlinearconv_pyccel
!........................................

end module mod_lh1i7w9c
