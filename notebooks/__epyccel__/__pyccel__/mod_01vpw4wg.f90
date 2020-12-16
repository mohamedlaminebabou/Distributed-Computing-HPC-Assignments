module mod_01vpw4wg

use ISO_C_BINDING

implicit none

contains

!........................................
function solve_1d_diff_pyccel(u, un, nt, nx, dt, dx, nu) result(Out_0001 &
      )

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(inout) :: u(0:)
  real(C_DOUBLE), intent(inout) :: un(0:)
  integer(C_LONG_LONG), value :: nt
  integer(C_LONG_LONG), value :: nx
  real(C_DOUBLE), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE), value :: nu
  integer(C_LONG_LONG) :: n
  integer(C_LONG_LONG) :: i

  do n = 0_C_LONG_LONG, nt-1_C_LONG_LONG, 1_C_LONG_LONG
    un(:) = u(:)
    do i = 1_C_LONG_LONG, nx - 1_C_LONG_LONG-1_C_LONG_LONG, &
      1_C_LONG_LONG
      u(i) = un(i) + nu * dt / dx ** 2_C_LONG_LONG * (un(i + &
      1_C_LONG_LONG) - 2_C_LONG_LONG * un(i) + un(i - 1_C_LONG_LONG))
    end do
  end do
  Out_0001 = 0_C_LONG_LONG
  return

end function solve_1d_diff_pyccel
!........................................

end module mod_01vpw4wg
