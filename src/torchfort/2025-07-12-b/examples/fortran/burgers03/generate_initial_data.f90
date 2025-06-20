program generate_initial_data
  use, intrinsic :: iso_fortran_env, only: real32
  implicit none

  integer, parameter :: N_0 = 400
  real(real32), parameter :: x_min = -1.0, x_max = 1.0
  real(real32) :: x0(N_0), t0(N_0), u0(N_0)
  integer :: i
  real(real32) :: pi

  pi = acos(-1.0)

  do i = 1, N_0
    x0(i) = x_min + (i - 1) * (x_max - x_min) / (N_0 - 1)
    t0(i) = 0.0
    u0(i) = -sin(pi * x0(i))
  end do

  open(unit=10, file='initial_condition.bin', form='unformatted', access='stream')
  write(10) x0
  write(10) t0
  write(10) u0
  close(10)

end program generate_initial_data
