program burgers03
  use torchfort
  use, intrinsic :: iso_c_binding, only: c_float, c_int64_t
  implicit none

  integer(c_int) :: istat
  integer :: device = 0 ! 0 for GPU, -1 for CPU

  ! Parameters from burgers1d.py
  integer, parameter :: N_f = 10000
  integer, parameter :: N_0 = 400
  integer, parameter :: N_b = 200
  real(c_float), parameter :: x_min = -1.0, x_max = 1.0
  real(c_float), parameter :: t_min = 0.0, t_max = 1.0
  real(c_float), parameter :: nu = 0.01 / 3.141592653589793

  ! Training data arrays
  real(c_float), allocatable :: X_f(:,:)
  real(c_float), allocatable :: x0_train(:,:), t0_train(:,:), u0_train(:,:)
  real(c_float), allocatable :: xb_left_train(:,:), tb_left_train(:,:), ub_left_train(:,:)
  real(c_float), allocatable :: xb_right_train(:,:), tb_right_train(:,:), ub_right_train(:,:)

  ! Tensor lists
  type(torchfort_tensor_list) :: input_tensors, label_tensors, extra_args_tensors

  ! Other variables
  integer :: i
  real(c_float) :: loss_val
  integer, parameter :: num_epochs = 1

  !--------------------------------------------------------------------
  ! Generate training data (mimicking Python script)
  !--------------------------------------------------------------------
  call generate_training_data()

  print *, "Successfully generated training data."
  print *, "Shape of X_f: ", shape(X_f)
  print *, "First row of X_f: ", X_f(1,:)
  call flush(6)

  !--------------------------------------------------------------------
  ! Initialize TorchFort and create model
  !--------------------------------------------------------------------
  istat = torchfort_create_model("burgers_model", "config_burgers_torchscript.yaml", device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_create_model failed with status: ", istat
    stop
  endif
  print *, "Model created successfully."
  call flush(6)

  !--------------------------------------------------------------------
  ! Training loop
  !--------------------------------------------------------------------
  print *, "Starting training loop..."
  call flush(6)

  do i = 1, num_epochs
    ! Create tensor lists for this epoch
    istat = torchfort_tensor_list_create(input_tensors)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_create(label_tensors) ! Empty for PINN
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_create(extra_args_tensors) ! Empty for this case
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    ! Add tensors to the input list
    istat = torchfort_tensor_list_add_tensor(input_tensors, X_f)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, x0_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, t0_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, u0_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, xb_left_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, tb_left_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, ub_left_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, xb_right_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, tb_right_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_add_tensor(input_tensors, ub_right_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    ! Add tensors to the label list
    istat = torchfort_tensor_list_add_tensor(label_tensors, u0_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    ! Add tensors to the label list
    istat = torchfort_tensor_list_add_tensor(label_tensors, u0_train)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    ! Perform a training step
    istat = torchfort_train_multiarg("burgers_model", input_tensors, label_tensors, loss_val)
    if (istat /= TORCHFORT_RESULT_SUCCESS) then
        print *, "ERROR: torchfort_train_multiarg failed in epoch", i, " with status: ", istat
        stop
    endif

    ! Destroy tensor lists for this epoch
    istat = torchfort_tensor_list_destroy(input_tensors)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_destroy(label_tensors)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_tensor_list_destroy(extra_args_tensors)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop

    if (mod(i, 100) == 0) then
      print *, "Epoch: ", i, " Loss: ", loss_val
      call flush(6)
    endif
  end do

  print *, "Training finished. Final loss: ", loss_val
  call flush(6)

  !--------------------------------------------------------------------
  ! Save the trained model
  !--------------------------------------------------------------------
  istat = torchfort_save_model("burgers_model", "burgers_model_trained.pt")
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
      print *, "ERROR: torchfort_save_model failed with status: ", istat
      stop
  endif
  print *, "Trained model saved to burgers_model_trained.pt"
  call flush(6)

contains

  subroutine generate_training_data()
    implicit none
    real(c_float) :: r1, r2
    integer :: i
    real(c_float), allocatable :: rand_nums(:,:)

    ! Allocate arrays
    allocate(X_f(2, N_f))
    allocate(x0_train(1, N_0), t0_train(1, N_0), u0_train(1, N_0))
    allocate(xb_left_train(1, N_b), tb_left_train(1, N_b), ub_left_train(1, N_b))
    allocate(xb_right_train(1, N_b), tb_right_train(1, N_b), ub_right_train(1, N_b))

    ! 1. Collocation points (X_f)
    call random_number(X_f)
    X_f(1, :) = X_f(1, :) * (x_max - x_min) + x_min
    X_f(2, :) = X_f(2, :) * (t_max - t_min) + t_min

    ! 2. Initial condition (t=0)
    do i = 1, N_0
        x0_train(1, i) = x_min + (i - 1) * (x_max - x_min) / (N_0 - 1)
    end do
    t0_train = 0.0
    u0_train = -sin(3.141592653589793 * x0_train)

    ! 3. Boundary conditions (x=-1 and x=1)
    do i = 1, N_b
        tb_left_train(1, i) = t_min + (i - 1) * (t_max - t_min) / (N_b - 1)
        tb_right_train(1, i) = tb_left_train(1, i)
    end do
    xb_left_train = x_min
    xb_right_train = x_max
    ub_left_train = 0.0
    ub_right_train = 0.0

  end subroutine generate_training_data

end program burgers03