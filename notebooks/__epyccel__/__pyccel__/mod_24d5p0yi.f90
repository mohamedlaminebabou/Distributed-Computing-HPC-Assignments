module mod_24d5p0yi

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

  L = dt / dx
  do j = 0_C_LONG_LONG, nt-1_C_LONG_LONG, 1_C_LONG_LONG
    un(:) = u(:)
    do i = 1_C_LONG_LONG, nx-1_C_LONG_LONG, 1_C_LONG_LONG
      un(i) = un(i) + L * un(i) * (un(i) - un(i - 1_C_LONG_LONG))
    end do
  end do
  Out_0001 = 0_C_LONG_LONG
  return

end function solve_1d_nonlinearconv_pyccel
!........................................

end module mod_24d5p0yi
