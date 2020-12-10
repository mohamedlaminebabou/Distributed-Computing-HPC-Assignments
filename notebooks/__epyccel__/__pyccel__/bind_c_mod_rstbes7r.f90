!........................................
function dot(n0_a, n1_a, a, n0_b, n1_b, b, n0_c, n1_c, c) bind(c) result &
      (Out_0001)

  use mod_rstbes7r, only: mod_dot => dot
  use ISO_C_BINDING
  implicit none

  integer(C_INT), value :: n0_a
  integer(C_INT), value :: n1_a
  real(C_DOUBLE), intent(in) :: a(0:n1_a-1,0:n0_a-1)
  integer(C_INT), value :: n0_b
  integer(C_INT), value :: n1_b
  real(C_DOUBLE), intent(in) :: b(0:n0_b-1,0:n1_b-1)
  integer(C_INT), value :: n0_c
  integer(C_INT), value :: n1_c
  real(C_DOUBLE), intent(inout) :: c(0:n1_c-1,0:n0_c-1)
  integer(C_LONG_LONG) :: Out_0001


  Out_0001 = mod_dot(a, b, c)

end function dot
!........................................
