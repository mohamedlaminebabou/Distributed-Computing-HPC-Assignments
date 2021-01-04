!........................................
function solve_2d_diff_pyccel(n0_u, n1_u, u, n0_un, n1_un, un, nt, dt, &
      dx, dy, nu) bind(c) result(Out_0001)

  use mod_l1cnyxeo, only: mod_solve_2d_diff_pyccel => &
      solve_2d_diff_pyccel
  use ISO_C_BINDING
  implicit none

  integer(C_INT), value :: n0_u
  integer(C_INT), value :: n1_u
  real(C_DOUBLE), intent(inout) :: u(0:n1_u-1,0:n0_u-1)
  integer(C_INT), value :: n0_un
  integer(C_INT), value :: n1_un
  real(C_DOUBLE), intent(inout) :: un(0:n1_un-1,0:n0_un-1)
  integer(C_LONG_LONG), value :: nt
  real(C_DOUBLE), value :: dt
  real(C_DOUBLE), value :: dx
  real(C_DOUBLE), value :: dy
  real(C_DOUBLE), value :: nu
  integer(C_LONG_LONG) :: Out_0001


  Out_0001 = mod_solve_2d_diff_pyccel(u, un, nt, dt, dx, dy, nu)

end function solve_2d_diff_pyccel
!........................................
