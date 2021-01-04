module mod_l1cnyxeo

use ISO_C_BINDING

implicit none

contains

!........................................
function solve_2d_diff_pyccel(u, un, nt, dt, dx, dy, nu) result(Out_0001 &
      )

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(inout) :: u(0:,0:)
  real(C_DOUBLE), intent(inout) :: un(0:,0:)
  integer(C_LONG_LONG), value :: nt
  real(C_DOUBLE), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE), value :: dy
  real(C_DOUBLE), value :: nu
  integer(C_LONG_LONG) :: row
  integer(C_LONG_LONG) :: col
  integer(C_LONG_LONG) :: j
  integer(C_LONG_LONG) :: ix
  integer(C_LONG_LONG) :: iy

  row = size(u, 2)
  col = size(u, 1)
  u(Int(0.5_C_DOUBLE / dx, 8):Int(1_C_LONG_LONG / dx + 1_C_LONG_LONG, 8 &
      ) - 1_C_LONG_LONG, Int(0.5_C_DOUBLE / dy, 8):Int(1_C_LONG_LONG / &
      dy + 1_C_LONG_LONG, 8) - 1_C_LONG_LONG) = 2_C_LONG_LONG
  do j = 0_C_LONG_LONG, nt + 1_C_LONG_LONG-1_C_LONG_LONG, 1_C_LONG_LONG
    do ix = 1_C_LONG_LONG, row-1_C_LONG_LONG, 1_C_LONG_LONG
      do iy = 1_C_LONG_LONG, col-1_C_LONG_LONG, 1_C_LONG_LONG
        un(iy, ix) = u(iy, ix) + nu * (dt / dx ** 2_C_LONG_LONG) * (u(iy &
      , ix + 1_C_LONG_LONG) - 2_C_LONG_LONG * u(iy, ix) + u(iy, ix - &
      1_C_LONG_LONG)) + nu * dt / dy ** 2_C_LONG_LONG * (u(iy + &
      1_C_LONG_LONG, ix) - 2_C_LONG_LONG * u(iy, ix) + u(iy - &
      1_C_LONG_LONG, ix))
      end do
    end do
    u(:, 0_C_LONG_LONG) = 1_C_LONG_LONG
    u(:, size(u, 1) - 1_C_LONG_LONG) = 1_C_LONG_LONG
    u(0_C_LONG_LONG, :) = 1_C_LONG_LONG
    u(size(u, 2) - 1_C_LONG_LONG, :) = 1_C_LONG_LONG
    !fill the update of u and v
  end do
  Out_0001 = 0_C_LONG_LONG
  return

end function solve_2d_diff_pyccel
!........................................

end module mod_l1cnyxeo
