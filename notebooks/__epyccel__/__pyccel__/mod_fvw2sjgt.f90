module mod_fvw2sjgt

use ISO_C_BINDING

implicit none

contains

!........................................
function solve_1d_linearconv_f90(u, un, nt, nx, dt, dx, c) result( &
      Out_0001)

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(inout) :: u(0:)
  real(C_DOUBLE), intent(inout) :: un(0:)
  integer(C_LONG_LONG), value :: nt
  integer(C_LONG_LONG), value :: nx
  real(C_DOUBLE), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE), value :: c
  integer(C_LONG_LONG) :: j
  integer(C_LONG_LONG) :: i

  do j = 0_C_LONG_LONG, nt-1_C_LONG_LONG, 1_C_LONG_LONG
    un(:) = u(:)
    do i = 0_C_LONG_LONG, nx - 1_C_LONG_LONG-1_C_LONG_LONG, &
      1_C_LONG_LONG
      u(i + 1_C_LONG_LONG) = (1_C_LONG_LONG - c * (dt / dx)) * un(i + &
      1_C_LONG_LONG) + c * (dt / dx) * un(i)
    end do
  end do
  Out_0001 = 0_C_LONG_LONG
  return

end function solve_1d_linearconv_f90
!........................................

end module mod_fvw2sjgt
