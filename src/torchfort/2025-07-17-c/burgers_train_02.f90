program burgers_train_02
  use, intrinsic :: iso_fortran_env, only: real32, real64
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
#ifdef _OPENACC
  use openacc
#endif
  use torchfort
  implicit none

  integer :: i, istat, j, k
  real(real32) :: loss_val
  character(len=256) :: configfile = "config_burgers.yaml"
  character(len=256) :: output_model_name = "burgers_model_trained.pt"
  character(len=256) :: output_checkpoint_dir = "checkpoint"
  integer :: ntrain_steps = 5000 ! Standardized for faster development
  integer :: model_device = 0

  ! command line arguments (simplified for now)
  logical :: skip_next
  character(len=256) :: arg

  ! Physical parameters and boundaries
  real(real32), parameter :: x_min = -1.0, x_max = 1.0  ! Spatial range
  real(real32), parameter :: t_min = 0.0, t_max = 1.0   ! Temporal range
  real(real32), parameter :: nu = 0.01 / 3.14159265358979323846_real32 ! Diffusion coefficient (using pi from Python)

  ! Number of sampled points for training
  integer, parameter :: N_f = 10000  ! Collocation points for PDE
  integer, parameter :: N_0 = 400    ! Initial condition points
  integer, parameter :: N_b = 200    ! Boundary condition points

  ! Total number of training points
  integer, parameter :: N_total = N_f + N_0 + 2*N_b

  ! Data arrays for training
  real(real32), allocatable :: X_f(:,:)
  real(real32), allocatable :: x0(:), t0(:), u0(:)
  real(real32), allocatable :: tb(:), xb_left(:), xb_right(:), ub_left(:), ub_right(:)

  ! Combined input and label tensors for TorchFort training
  real(real32), allocatable :: input_data(:,:)
  real(real32), allocatable :: label_data(:,:)

  ! Grid dimensions for inference visualization
  integer, parameter :: N_x_inf = 256, N_t_inf = 100

  ! Data arrays for inference
  real(real32), allocatable :: x_inf(:), t_inf(:)
  real(real32), allocatable :: XT_tensor_inf(:,:)
  real(real32), allocatable :: u_pred_inf(:,:)
  character(len=256) :: inference_output_filename = "fortran_trained_u_pred_02.bin"

  ! read command line arguments
  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--configfile')
        call get_command_argument(i+1, arg)
        read(arg, *) configfile
        skip_next = .true.
      case('--output_model_name')
        call get_command_argument(i+1, arg)
        read(arg, *) output_model_name
        skip_next = .true.
      case('--output_checkpoint_dir')
        call get_command_argument(i+1, arg)
        read(arg, *) output_checkpoint_dir
        skip_next = .true.
      case('--ntrain_steps')
        call get_command_argument(i+1, arg)
        read(arg, *) ntrain_steps
        skip_next = .true.
      case('--train_device')
        call get_command_argument(i+1, arg)
        read(arg, *) model_device
        if (model_device < -1) then
          print*, "Invalid train device type argument."
          call exit(1)
        endif
        skip_next = .true.
      case default
        print*, "Unknown argument: ", trim(arg)
        call exit(1)
    end select
  end do

  print*, "Run settings:"
  print*, "\tconfigfile: ", trim(configfile)
  if (model_device == TORCHFORT_DEVICE_CPU) then
    print*, "\ttrain_device: cpu"
  else
    print*, "\ttrain_device: gpu ", model_device
  endif
  print*, "\toutput_model_name: ", trim(output_model_name)
  print*, "\toutput_checkpoint_dir: ", trim(output_checkpoint_dir)
  print*, "\tntrain_steps: ", ntrain_steps
  print*

  ! set torch benchmark mode
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! setup the model
  istat = torchfort_create_model("mymodel", configfile, model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! Allocate training data arrays
  allocate(X_f(2, N_f))
  allocate(x0(N_0), t0(N_0), u0(N_0))
  allocate(tb(N_b), xb_left(N_b), xb_right(N_b), ub_left(N_b), ub_right(N_b))

  ! Allocate combined input and label tensors for training
  allocate(input_data(2, N_total))  ! (x, t) for all points
  allocate(label_data(1, N_total))  ! u_pred or 0 for all points

  ! Seed the random number generator for reproducibility
  call random_seed()

  ! Generate collocation points for PDE (X_f)
  do i = 1, N_f
    call random_number(X_f(1, i))
    call random_number(X_f(2, i))
    X_f(1, i) = X_f(1, i) * (x_max - x_min) + x_min  ! Normalize x to [-1, 1]
    X_f(2, i) = X_f(2, i) * (t_max - t_min) + t_min  ! Normalize t to [0, 1]
  end do

  ! Initial condition u(x, 0) = -sin(pi * x)
  do i = 1, N_0
    x0(i) = x_min + (i-1) * (x_max - x_min) / (N_0 - 1)
    t0(i) = 0.0_real32
    u0(i) = -sin(nu * 3.14159265358979323846_real32 * x0(i)) ! nu * pi * x0
  end do

  ! Boundary conditions u(-1,t) = 0 and u(1,t) = 0
  do i = 1, N_b
    tb(i) = t_min + (i-1) * (t_max - t_min) / (N_b - 1)
    xb_left(i) = x_min
    xb_right(i) = x_max
    ub_left(i) = 0.0_real32
    ub_right(i) = 0.0_real32
  end do

  ! Combine all training data into input_data and label_data
  ! PDE points
  j = 1
  do i = 1, N_f
    input_data(1, j) = X_f(1, i)
    input_data(2, j) = X_f(2, i)
    label_data(1, j) = 0.0_real32 ! Target for PDE residual is 0
    j = j + 1
  end do

  ! Initial condition points
  do i = 1, N_0
    input_data(1, j) = x0(i)
    input_data(2, j) = t0(i)
    label_data(1, j) = u0(i) ! Target for initial condition is u0
    j = j + 1
  end do

  ! Left boundary condition points
  do i = 1, N_b
    input_data(1, j) = xb_left(i)
    input_data(2, j) = tb(i)
    label_data(1, j) = ub_left(i) ! Target for boundary condition is 0
    j = j + 1
  end do

  ! Right boundary condition points
  do i = 1, N_b
    input_data(1, j) = xb_right(i)
    input_data(2, j) = tb(i)
    label_data(1, j) = ub_right(i) ! Target for boundary condition is 0
    j = j + 1
  end do

  print*, "Starting training loop..."
  do i = 1, ntrain_steps
    ! Call torchfort_train with combined data
    istat = torchfort_train("mymodel", input_data, label_data, loss_val)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    if (mod(i, 100) == 0) then
      print*, "Training step: ", i, " Loss: ", loss_val
    endif
  end do
  print*, "Training finished. Final loss: ", loss_val

  print*, "saving model and writing checkpoint..."
  istat = torchfort_save_model("mymodel", output_model_name)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  istat = torchfort_save_checkpoint("mymodel", output_checkpoint_dir)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! --- Inference Section (Added for Validation) ---
  print*, "\nStarting inference for validation..."

  ! Allocate inference data arrays
  allocate(x_inf(N_x_inf), t_inf(N_t_inf))
  allocate(XT_tensor_inf(2, N_x_inf * N_t_inf)) ! (x, t) for all points
  allocate(u_pred_inf(1, N_x_inf * N_t_inf))    ! u_pred for all points

  ! Generate grid for visualization (meshgrid equivalent)
  do i = 1, N_x_inf
    x_inf(i) = x_min + (i-1) * (x_max - x_min) / (N_x_inf - 1)
  end do

  do i = 1, N_t_inf
    t_inf(i) = t_min + (i-1) * (t_max - t_min) / (N_t_inf - 1)
  end do

  ! Populate XT_tensor_inf (hstack equivalent)
  j = 1
  do i = 1, N_t_inf
    do k = 1, N_x_inf
      XT_tensor_inf(1, j) = x_inf(k)
      XT_tensor_inf(2, j) = t_inf(i)
      j = j + 1
    end do
  end do

  print*, "Performing inference..."
  ! Perform inference
  istat = torchfort_inference("mymodel", XT_tensor_inf, u_pred_inf)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  print*, "Inference complete."

  ! Save results to a binary file
  open(unit=10, file=inference_output_filename, form='binary', status='replace')
  write(10) N_x_inf, N_t_inf
  write(10) x_inf
  write(10) t_inf
  write(10) u_pred_inf
  close(10)
  print*, "Inference results saved to ", trim(inference_output_filename)

  ! Deallocate all arrays
  deallocate(X_f, x0, t0, u0, tb, xb_left, xb_right, ub_left, ub_right)
  deallocate(input_data, label_data)
  deallocate(x_inf, t_inf, XT_tensor_inf, u_pred_inf)

end program burgers_train_02
