program compare_cpu_gpu
  use openacc
  implicit none
  integer :: i, j, n
  real :: x, start_cpu, finish_cpu, start_gpu, finish_gpu

  n = 10**9
  x = 0.0

  ! Medindo tempo na CPU
  call cpu_time(start_cpu)
  do j = 1, 5
  do i = 1, n
    x = x + sqrt(real(i))
  end do
  write(*, '(A)', advance='no') '.'
  end do
  call cpu_time(finish_cpu)

  print *, "Tempo de execução na CPU:", finish_cpu - start_cpu, "segundos"

  x = 0.0  ! Resetando variável

  ! Medindo tempo na GPU
  call cpu_time(start_gpu)
  do j = 1, 5
  !$acc parallel loop
  do i = 1, n
    x = x + sqrt(real(i))
  end do
  !$acc end parallel loop
  write(*, '(A)', advance='no') '.'
  end do
  call cpu_time(finish_gpu)

  print *, "Tempo de execução na GPU:", finish_gpu - start_gpu, "segundos"

end program compare_cpu_gpu
