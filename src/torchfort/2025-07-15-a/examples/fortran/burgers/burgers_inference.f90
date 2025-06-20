program burgers_inference
  use torchfort
  use, intrinsic :: iso_c_binding, only: c_float, c_int64_t
  implicit none

  integer(c_int) :: istat
  integer :: device = 0 ! 0 for GPU, -1 for CPU
  integer :: i, j
  
  ! Parameters from burgers1d.py for evaluation grid
  real(c_float), parameter :: x_min = -1.0_c_float
  real(c_float), parameter :: x_max = 1.0_c_float
  real(c_float), parameter :: t_min = 0.0_c_float
  real(c_float), parameter :: t_max = 1.0_c_float
  integer, parameter :: N_x = 256
  integer, parameter :: N_t = 100
  integer, parameter :: total_points = N_x * N_t

  ! Declare arrays for input and output
  real(c_float), allocatable, contiguous, target :: XT_tensor(:,:)
  real(c_float), allocatable, contiguous, target :: u_pred_flat(:,:)
  real(c_float), allocatable :: x_grid(:), t_grid(:)
  real(c_float), allocatable :: u_pred_reshaped(:,:)

  ! TorchFort tensor lists
  type(torchfort_tensor_list) :: input_tensors
  type(torchfort_tensor_list) :: output_tensors

  !--------------------------------------------------------------------
  ! Generate evaluation grid (matching burgers1d.py)
  !--------------------------------------------------------------------
  allocate(x_grid(N_x))
  allocate(t_grid(N_t))
  allocate(XT_tensor(2, total_points))
  allocate(u_pred_flat(1, total_points))
  allocate(u_pred_reshaped(N_x, N_t))

  

  ! Populate x_grid and t_grid
  do i = 1, N_x
    x_grid(i) = x_min + (real(i-1, c_float) / real(N_x-1, c_float)) * (x_max - x_min)
  end do

  do i = 1, N_t
    t_grid(i) = t_min + (real(i-1, c_float) / real(N_t-1, c_float)) * (t_max - t_min)
  end do

  ! Populate XT_tensor (features, total_points) - original Fortran layout
  do j = 1, N_t
    do i = 1, N_x
      XT_tensor(1, (j-1)*N_x + i) = x_grid(i)
      XT_tensor(2, (j-1)*N_x + i) = t_grid(j)
    end do
  end do

  print *, "Successfully generated evaluation grid."
  call flush(6)

  !--------------------------------------------------------------------
  ! Initialize TorchFort and load model
  !--------------------------------------------------------------------
  print *, "Current working directory before model creation:"
  call system('pwd')
  call flush(6)
  print *, "Listing burgers_model.pt:"
  call system('ls -l burgers_model.pt')
  call flush(6)
  istat = torchfort_create_model("burgers_inference_model", "config_burgers_inference_architecture.yaml", device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_create_model failed with status: ", istat
    stop
  endif
  print *, "Inference model created successfully from architecture config."
  call flush(6)

  ! Load the trained model parameters
  istat = torchfort_load_model("burgers_inference_model", "burgers_model.pt")
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_load_model failed with status: ", istat
    stop
  endif
  print *, "Trained model loaded successfully."
  call flush(6)
  

  

  

  

  ! Save XT_tensor_transposed to a binary file for debugging
  open(unit=11, file="fortran_xt_tensor.bin", form="binary", access="sequential", status="replace")
  write(11) XT_tensor
  close(11)
  print *, "Fortran XT_tensor saved to fortran_xt_tensor.bin"
  call flush(6)

  !--------------------------------------------------------------------
  ! Create and populate tensor lists for inference
  !--------------------------------------------------------------------
  istat = torchfort_tensor_list_create(input_tensors)
  istat = torchfort_tensor_list_add_tensor(input_tensors, XT_tensor)

  istat = torchfort_tensor_list_create(output_tensors)
  istat = torchfort_tensor_list_add_tensor(output_tensors, u_pred_flat)

  !--------------------------------------------------------------------
  ! Perform inference
  !--------------------------------------------------------------------
  print *, "Starting inference..."
  call flush(6)

  istat = torchfort_inference("burgers_inference_model", XT_tensor, u_pred_flat)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_inference_multiarg failed with status: ", istat
    stop
  endif
  print *, "Inference complete."
  call flush(6)

  

  ! Reshape u_pred_flat to u_pred_reshaped (N_t, N_x)
  do j = 1, N_t
    do i = 1, N_x
      u_pred_reshaped(i, j) = u_pred_flat(1, (j-1)*N_x + i)
    end do
  end do

  !--------------------------------------------------------------------
  ! Save results to a binary file
  !--------------------------------------------------------------------
  open(unit=10, file="burgers1d_fortran_inference_results.bin", &
       form="binary", access="sequential", status="replace")

  write(10) N_x, N_t
  write(10) x_grid
  write(10) t_grid
  write(10) u_pred_reshaped

  close(10)
  print *, "Fortran inference results saved to examples/fortran/burgers/burgers1d_fortran_inference_results.bin"
  call flush(6)

  !--------------------------------------------------------------------
  ! Destroy tensor lists
  !--------------------------------------------------------------------
  istat = torchfort_tensor_list_destroy(input_tensors)
  istat = torchfort_tensor_list_destroy(output_tensors)

end program burgers_inference
