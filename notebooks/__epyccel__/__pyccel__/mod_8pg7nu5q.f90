module mod_8pg7nu5q

use ISO_C_BINDING

implicit none

contains

!........................................
function solve_2d_nonlinearconv_pyccel(u, un, v, vn, nt, dt, dx, dy, c) &
      result(Out_0001)

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(inout) :: u(0:,0:)
  real(C_DOUBLE), intent(inout) :: un(0:,0:)
  real(C_DOUBLE), intent(inout) :: v(0:,0:)
  real(C_DOUBLE), intent(inout) :: vn(0:,0:)
  integer(C_LONG_LONG), value :: nt
  real(C_DOUBLE), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE), value :: dy
  real(C_DOUBLE), value :: c
  integer(C_LONG_LONG) :: row
  integer(C_LONG_LONG) :: col
  integer(C_LONG_LONG) :: it
  integer(C_LONG_LONG) :: i
  integer(C_LONG_LONG) :: j

  !##Assign initial conditions
  !#set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
  u(Int(0.5_C_DOUBLE / dx, 8):Int(1_C_LONG_LONG / dx + 1_C_LONG_LONG, 8 &
      ) - 1_C_LONG_LONG, Int(0.5_C_DOUBLE / dy, 8):Int(1_C_LONG_LONG / &
      dy + 1_C_LONG_LONG, 8) - 1_C_LONG_LONG) = 2_C_LONG_LONG
  !#set hat function I.C. : v(.5<=x<=1 && .5<=y<=1 ) is 2
  v(Int(0.5_C_DOUBLE / dx, 8):Int(1_C_LONG_LONG / dx + 1_C_LONG_LONG, 8 &
      ) - 1_C_LONG_LONG, Int(0.5_C_DOUBLE / dy, 8):Int(1_C_LONG_LONG / &
      dy + 1_C_LONG_LONG, 8) - 1_C_LONG_LONG) = 2_C_LONG_LONG
  row = size(u, 2)
  col = size(u, 1)
  do it = 0_C_LONG_LONG, nt-1_C_LONG_LONG, 1_C_LONG_LONG
    un(:, :) = u(:, :)
    vn(:, :) = v(:, :)
    do i = 1_C_LONG_LONG, row-1_C_LONG_LONG, 1_C_LONG_LONG
      do j = 1_C_LONG_LONG, col-1_C_LONG_LONG, 1_C_LONG_LONG
        u(j, i) = un(j, i) - un(j, i) * (dt / dx) * (un(j, i) - un(j, i &
      - 1_C_LONG_LONG)) - vn(j, i) * (dt / dy) * (un(j, i) - un(j - &
      1_C_LONG_LONG, i))
        v(j, i) = vn(j, i) - un(j, i) * (dt / dx) * (vn(j, i) - vn(j, i &
      - 1_C_LONG_LONG)) - vn(j, i) * (dt / dy) * (vn(j, i) - vn(j - &
      1_C_LONG_LONG, i))
      end do
    end do
    u(:, 0_C_LONG_LONG) = 1_C_LONG_LONG
    u(:, size(u, 1) - 1_C_LONG_LONG) = 1_C_LONG_LONG
    u(0_C_LONG_LONG, :) = 1_C_LONG_LONG
    u(size(u, 2) - 1_C_LONG_LONG, :) = 1_C_LONG_LONG
    v(:, 0_C_LONG_LONG) = 1_C_LONG_LONG
    v(:, size(v, 1) - 1_C_LONG_LONG) = 1_C_LONG_LONG
    v(0_C_LONG_LONG, :) = 1_C_LONG_LONG
    v(size(v, 2) - 1_C_LONG_LONG, :) = 1_C_LONG_LONG
  end do
  Out_0001 = 0_C_LONG_LONG
  return

end function solve_2d_nonlinearconv_pyccel
!........................................

end module mod_8pg7nu5q
