program burgers_inference
  use, intrinsic :: iso_fortran_env, only: real32
  use torchfort
  implicit none

  integer :: istat
  integer :: model_device = 0

  ! Physics and data parameters
  real(real32), parameter :: x_min = -1.0, x_max = 1.0
  real(real32), parameter :: t_min = 0.0, t_max = 1.0
  integer, parameter :: N_x = 256, N_t = 100
  real(real32), parameter :: pi = acos(-1.0)

  ! Data arrays for inference
  real(real32) :: X(N_x, N_t), T(N_x, N_t)
  real(real32) :: XT(2, N_x * N_t)
  real(real32), allocatable :: u_pred(:,:)

  integer :: i, j, k

  ! Generate grid for inference
  do j = 1, N_t
    do i = 1, N_x
      X(i, j) = x_min + (i - 1) * (x_max - x_min) / real(N_x - 1)
      T(i, j) = t_min + (j - 1) * (t_max - t_min) / real(N_t - 1)
      XT(1, (j-1)*N_x + i) = X(i, j)
      XT(2, (j-1)*N_x + i) = T(i, j)
    end do
  end do

  ! Setup Model
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to set cudnn benchmark mode"

  istat = torchfort_create_model("mymodel", "burgers_inference_model.pt", model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to create model"

  ! Perform inference
  allocate(u_pred(1, N_x * N_t))
  istat = torchfort_inference("mymodel", XT, u_pred)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Inference failed"

  ! Reshape u_pred to (N_t, N_x) for saving
  ! Note: Fortran stores column-major, so direct reshape might need care
  ! For saving to text, we can write row by row.
  open(unit=10, file="fortran_u_pred.txt", status="replace")
  do j = 1, N_t
    do i = 1, N_x
      ! Assuming u_pred is (1, N_x * N_t) and contains flattened output
      ! Need to map back to (N_t, N_x) row-major for comparison with Python
      ! Fortran's XT is (2, N_x * N_t) where XT(1,:) is x and XT(2,:) is t
      ! The output u_pred(1, k) corresponds to XT(1,k), XT(2,k)
      ! To get u_pred(t_idx, x_idx) for Python comparison, we need to find k
      ! k = (t_idx-1)*N_x + x_idx
      write(10, fmt='(256(F15.8,1x))') u_pred(1, (j-1)*N_x + 1 : (j-1)*N_x + N_x)
    end do
  end do
  close(10)
  print *, "Fortran inference results saved to fortran_u_pred.txt"

  deallocate(u_pred)

end program burgers_inference
