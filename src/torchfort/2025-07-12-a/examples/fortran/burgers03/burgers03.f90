program burgers03
  use, intrinsic :: iso_fortran_env, only: real32
#ifdef _OPENACC
  use openacc
#endif
  use torchfort
  use simulation
  implicit none

  ! Physical parameters
  real(real32), parameter :: x_min = -1.0, x_max = 1.0
  real(real32), parameter :: t_min = 0.0, t_max = 1.0
  real(real32), parameter :: nu = 0.01 / acos(-1.0)

  ! Training parameters
  integer, parameter :: N_f = 10000
  integer, parameter :: N_0 = 400
  integer, parameter :: N_b = 200
  integer, parameter :: num_epochs = 5000

  ! TorchFort variables
  type(torchfort_tensor_list) :: input_tensors, label_tensors, extra_args_tensors
  integer :: istat
  real(real32) :: loss_val
  integer :: model_device = 0
  integer :: simulation_device = 0
#ifdef _OPENACC
  integer(acc_device_kind) :: dev_type
#endif
  integer :: train_step_ckpt = 0

  ! Data arrays
  real(real32), allocatable :: X_f(:,:), x0_t0(:,:), u0(:,:), xb_left_tb(:,:), xb_right_tb(:,:), u(:,:), u_div(:,:)
  integer :: i
  integer :: n = 32
  integer :: batch_size = 16
  real(real32) :: dt = 0.01
  real(real32) :: a(2) = [1.0, 0.789]

  ! Minimal command line argument parsing to mimic train.f90
  logical :: skip_next
  character(len=256) :: arg

  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    ! No actual arguments to process, just mimic the loop structure
  end do

  ! 1. Allocate and Generate Data
  print *, "Allocating arrays..."
  call flush(6)
  allocate(X_f(N_f, 2))
  allocate(x0_t0(N_0, 2), u0(N_0, 1))
  allocate(xb_left_tb(N_b, 2), xb_right_tb(N_b, 2))
  print *, "Arrays allocated."
  call flush(6)

  ! Allocate simulation data
  allocate(u(n, n))
  allocate(u_div(n, n))

  print *, "Generating training data..."
  call flush(6)
  call init_simulation(n, dt, a, real(train_step_ckpt)*real(batch_size)*dt, 0, 1, simulation_device)
  call random_number(X_f)
  X_f(:, 1) = X_f(:, 1) * (x_max - x_min) + x_min
  X_f(:, 2) = X_f(:, 2) * (t_max - t_min) + t_min

  do i = 1, N_0
    x0_t0(i, 1) = x_min + (i - 1) * (x_max - x_min) / real(N_0 - 1)
    x0_t0(i, 2) = 0.0
    u0(i, 1) = -sin(acos(-1.0) * x0_t0(i, 1))
  end do

  do i = 1, N_b
    xb_left_tb(i, 2) = t_min + (i - 1) * (t_max - t_min) / real(N_b - 1)
    xb_left_tb(i, 1) = x_min
    xb_right_tb(i, 2) = t_min + (i - 1) * (t_max - t_min) / real(N_b - 1)
    xb_right_tb(i, 1) = x_max
  end do
  print *, "Data generation complete."
  call flush(6)
  print *, "Shape of X_f:", shape(X_f)
  call flush(6)
  print *, "First row of X_f:", X_f(1,:)
  call flush(6)
  print *, "Shape of x0_t0:", shape(x0_t0)
  call flush(6)
  print *, "First row of x0_t0:", x0_t0(1,:)
  call flush(6)
  print *, "Shape of u0:", shape(u0)
  call flush(6)
  print *, "First value of u0:", u0(1,1)
  call flush(6)
  print *, "Shape of xb_left_tb:", shape(xb_left_tb)
  call flush(6)
  print *, "First row of xb_left_tb:", xb_left_tb(1,:)
  call flush(6)
  print *, "Shape of xb_right_tb:", shape(xb_right_tb)
  call flush(6)
  print *, "First row of xb_right_tb:", xb_right_tb(1,:)
  call flush(6)

  ! 2. Setup Model
  print *, "Creating model from config_mlp_native.yaml..."
  call flush(6)
#ifdef _OPENACC
  if (simulation_device >= 0) then
    dev_type = acc_get_device_type()
    call acc_set_device_num(simulation_device, dev_type)
    call acc_init(dev_type)
  endif
#endif
  
  ! set torch benchmark mode
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to set cudnn benchmark mode"

  print *, "Calling torchfort_create_model..."
  call flush(6)
  istat = torchfort_create_model("mymodel", "config_mlp_native.yaml", model_device)
  print *, "torchfort_create_model returned with status: ", istat
  call flush(6)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to create model"
  print *, "Model created successfully."
  call flush(6)

  ! 3. Training Loop
  print *, "Starting training loop for ", num_epochs, " epochs..."
  call flush(6)
  do i = 1, num_epochs
    print *, "Calling torchfort_tensor_list_create for input_tensors..."
    call flush(6)
    istat = torchfort_tensor_list_create(input_tensors)
    print *, "torchfort_tensor_list_create for input_tensors returned with status: ", istat
    call flush(6)
    if (istat /= 0) stop "Failed to create input_tensors"
    print *, "Calling torchfort_tensor_list_create for label_tensors..."
    call flush(6)
    istat = torchfort_tensor_list_create(label_tensors)
    print *, "torchfort_tensor_list_create for label_tensors returned with status: ", istat
    call flush(6)
    if (istat /= 0) stop "Failed to create label_tensors"
    print *, "Calling torchfort_tensor_list_create for extra_args_tensors..."
    call flush(6)
    istat = torchfort_tensor_list_create(extra_args_tensors)
    print *, "torchfort_tensor_list_create for extra_args_tensors returned with status: ", istat
    call flush(6)
    if (istat /= 0) stop "Failed to create extra_args_tensors"

    print *, "Calling torchfort_tensor_list_add_tensor for X_f..."
    call flush(6)
    istat = torchfort_tensor_list_add_tensor(input_tensors, X_f)
    print *, "torchfort_tensor_list_add_tensor for X_f returned with status: ", istat
    call flush(6)
    print *, "Calling torchfort_tensor_list_add_tensor for x0_t0..."
    call flush(6)
    istat = torchfort_tensor_list_add_tensor(input_tensors, x0_t0)
    print *, "torchfort_tensor_list_add_tensor for x0_t0 returned with status: ", istat
    call flush(6)
    print *, "Calling torchfort_tensor_list_add_tensor for u0..."
    call flush(6)
    istat = torchfort_tensor_list_add_tensor(input_tensors, u0)
    print *, "torchfort_tensor_list_add_tensor for u0 returned with status: ", istat
    call flush(6)
    print *, "Calling torchfort_tensor_list_add_tensor for xb_left_tb..."
    call flush(6)
    istat = torchfort_tensor_list_add_tensor(input_tensors, xb_left_tb)
    print *, "torchfort_tensor_list_add_tensor for xb_left_tb returned with status: ", istat
    call flush(6)
    print *, "Calling torchfort_tensor_list_add_tensor for xb_right_tb..."
    call flush(6)
    istat = torchfort_tensor_list_add_tensor(input_tensors, xb_right_tb)
    print *, "torchfort_tensor_list_add_tensor for xb_right_tb returned with status: ", istat
    call flush(6)

    print *, "Calling torchfort_train_multiarg..."
    call flush(6)
    istat = torchfort_train_multiarg("mymodel", input_tensors, label_tensors, loss_val, extra_args_tensors)
    print *, "torchfort_train_multiarg returned with status: ", istat
    call flush(6)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Training step failed"

    if (mod(i, 500) == 0 .or. i == 1) then
      print *, "Epoch ", i, "/", num_epochs, ", Loss: ", loss_val
      call flush(6)
    end if

    print *, "Calling torchfort_tensor_list_destroy for input_tensors..."
    call flush(6)
    istat = torchfort_tensor_list_destroy(input_tensors)
    print *, "torchfort_tensor_list_destroy for input_tensors returned with status: ", istat
    call flush(6)
    print *, "Calling torchfort_tensor_list_destroy for label_tensors..."
    call flush(6)
    istat = torchfort_tensor_list_destroy(label_tensors)
    print *, "torchfort_tensor_list_destroy for label_tensors returned with status: ", istat
    call flush(6)
    print *, "Calling torchfort_tensor_list_destroy for extra_args_tensors..."
    call flush(6)
    istat = torchfort_tensor_list_destroy(extra_args_tensors)
    print *, "torchfort_tensor_list_destroy for extra_args_tensors returned with status: ", istat
    call flush(6)
  end do
  print *, "Training complete!"
  call flush(6)

  ! 4. Save Model
  print *, "Saving trained model to burgers_model_trained.pt..."
  call flush(6)
  istat = torchfort_save_model("mymodel", "burgers_model_trained.pt")
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to save model"
  print *, "Model saved successfully."
  call flush(6)

end program burgers03