program burgers_inference
  use, intrinsic :: iso_fortran_env, only: real32
  use torchfort
  implicit none

  ! Variable Declarations
  integer :: i, istat, ierr
  integer :: model_device = 0
  integer, parameter :: N_x = 256, N_t = 100
  real(real32), parameter :: x_min = -1.0, x_max = 1.0
  real(real32), parameter :: t_min = 0.0, t_max = 1.0
  real(real32), allocatable :: x_grid(:), t_grid(:)
  real(real32), allocatable :: XT_tensor(:,:)
  real(real32), allocatable :: u_pred(:) ! Declared as 1D to match flattened PyTorch output
  type(torchfort_tensor_list) :: inference_inputs
  type(torchfort_tensor_list) :: inference_outputs
  character(len=256) :: output_filename

  ! 1. Generate grid data
  allocate(x_grid(N_x))
  allocate(t_grid(N_t))

  do i = 1, N_x
    x_grid(i) = x_min + (real(i) - 1.0) * (x_max - x_min) / real(N_x - 1)
  end do

  do i = 1, N_t
    t_grid(i) = t_min + (real(i) - 1.0) * (t_max - t_min) / real(N_t - 1)
  end do

  ! Create XT_tensor for inference (flattened meshgrid)
  ! Fortran is column-major, so XT_tensor(2, N_x * N_t)
  allocate(XT_tensor(2, N_x * N_t))
  do i = 1, N_t
    do istat = 1, N_x
      XT_tensor(1, (i - 1) * N_x + istat) = x_grid(istat)
      XT_tensor(2, (i - 1) * N_x + istat) = t_grid(i)
    end do
  end do

  print *, "Successfully generated inference grid data."

  ! 2. Load the trained model
  ! Use the config file which points to the trained inference network
  istat = torchfort_create_model("inference_model", "/torchfort/examples/fortran/burgers03/burgers_config.yaml", model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to create inference model"

  print *, "Successfully loaded trained model for inference."

  ! 3. Perform Inference
  ! The inference model expects a single input tensor (x,t) pairs.
  istat = torchfort_tensor_list_create(inference_inputs)
  if (istat /= 0) stop "Failed to create inference_inputs"

  istat = torchfort_tensor_list_add_tensor(inference_inputs, XT_tensor)
  if (istat /= 0) stop "Failed to add tensor to inference_inputs"

  istat = torchfort_tensor_list_create(inference_outputs)
  if (istat /= 0) stop "Failed to create inference_outputs"

  ! Allocate u_pred before inference, as TorchFort will write directly into it
  allocate(u_pred(N_x * N_t))

  ! Add u_pred to the output list for TorchFort to populate
  istat = torchfort_tensor_list_add_tensor(inference_outputs, u_pred)
  if (istat /= 0) stop "Failed to add u_pred to inference_outputs"

  istat = torchfort_inference_multiarg("inference_model", inference_inputs, inference_outputs)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Inference failed"

  print *, "Successfully performed inference."

  ! 4. Save results to a binary file (similar to Python's output)
  output_filename = "burgers1d_fortran_inference_results.bin"
  open(newunit=ierr, file=output_filename, form='unformatted', status='replace', iostat=istat)
  if (istat /= 0) then
    print *, "Error opening file: ", trim(output_filename)
    stop
  end if

  write(ierr) N_x, N_t
  write(ierr) x_grid
  write(ierr) t_grid
  write(ierr) u_pred ! u_pred is already 1D, consistent with Python's flattened output

  close(ierr)
  print *, "Inference results saved to ", trim(output_filename)

  ! Clean up
  istat = torchfort_tensor_list_destroy(inference_inputs)
  istat = torchfort_tensor_list_destroy(inference_outputs)
  ! Note: torchfort_destroy_model is not explicitly called, assuming library handles cleanup.

  deallocate(x_grid)
  deallocate(t_grid)
  deallocate(XT_tensor)
  deallocate(u_pred)

end program burgers_inference
