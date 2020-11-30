module mod_8e3i6g4r

use ISO_C_BINDING

implicit none

contains

!........................................
function dot(a, b, c) result(Out_0001)

  implicit none

  integer(C_LONG_LONG) :: Out_0001
  real(C_DOUBLE), intent(in) :: a(0:,0:)
  real(C_DOUBLE), intent(in) :: b(0:,0:)
  real(C_DOUBLE), intent(inout) :: c(0:,0:)
  integer(C_LONG_LONG) :: m
  integer(C_LONG_LONG) :: p
  integer(C_LONG_LONG) :: q
  integer(C_LONG_LONG) :: n
  integer(C_LONG_LONG) :: r
  integer(C_LONG_LONG) :: s
  integer(C_LONG_LONG) :: i
  integer(C_LONG_LONG) :: j
  integer(C_LONG_LONG) :: k

  m = size(a, 2)
  p = size(a, 1)
  q = size(b, 1)
  n = size(b, 2)
  r = size(c, 2)
  s = size(c, 1)
  if (p /= q .or. m /= r .or. n /= s) then
    Out_0001 = -1_C_LONG_LONG
    return
  end if
  !$omp parallel
  !$omp do schedule(runtime)
  do i = 0_C_LONG_LONG, m-1_C_LONG_LONG, 1_C_LONG_LONG
    do j = 0_C_LONG_LONG, n-1_C_LONG_LONG, 1_C_LONG_LONG
      c(j, i) = 0.0_C_DOUBLE
      do k = 0_C_LONG_LONG, p-1_C_LONG_LONG, 1_C_LONG_LONG
        c(j, i) = c(j, i) + a(k, i) * b(k, j)
        [!$omp end do  
        , !$omp end parallel  
        ]end do
      end do
    end do
    Out_0001 = 0_C_LONG_LONG
    return

  end function dot
  !........................................

  end module mod_8e3i6g4r
