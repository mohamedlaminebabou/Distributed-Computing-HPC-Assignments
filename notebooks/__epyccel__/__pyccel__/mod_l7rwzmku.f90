module mod_l7rwzmku

use ISO_C_BINDING

implicit none

contains

!........................................
function solve_2d_linearconv_pyccel(u, un, nt, dt, dx, dy, c) result( &
      Out_0001)

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(inout) :: u(0:,0:)
  real(C_DOUBLE), intent(inout) :: un(0:,0:)
  integer(C_LONG_LONG), value :: nt
  integer(C_LONG_LONG), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE), value :: dy
  real(C_DOUBLE), value :: c
  integer(C_LONG_LONG) :: n
  integer(C_LONG_LONG) :: row
  integer(C_LONG_LONG) :: col
  integer(C_LONG_LONG) :: j
  integer(C_LONG_LONG) :: i

  do n = 0_C_LONG_LONG, nt + 1_C_LONG_LONG-1_C_LONG_LONG, 1_C_LONG_LONG
    un(:, :) = u(:, :)
    row = size(u, 2)
    col = size(u, 1)
    do j = 1_C_LONG_LONG, row-1_C_LONG_LONG, 1_C_LONG_LONG
      do i = 1_C_LONG_LONG, col-1_C_LONG_LONG, 1_C_LONG_LONG
        u(i, j) = un(i, j) - c * dt / dx * (un(i, j) - un(i - &
      1_C_LONG_LONG, j)) - c * dt / dy * (un(i, j) - un(i, j - &
      1_C_LONG_LONG))
        u(:, 0_C_LONG_LONG) = 1_C_LONG_LONG
        u(:, size(u, 1) - 1_C_LONG_LONG) = 1_C_LONG_LONG
        u(0_C_LONG_LONG, :) = 1_C_LONG_LONG
        u(size(u, 2) - 1_C_LONG_LONG, :) = 1_C_LONG_LONG
      end do
    end do
    Out_0001 = 0_C_LONG_LONG
    return
  end do

end function solve_2d_linearconv_pyccel
!........................................

end module mod_l7rwzmku
