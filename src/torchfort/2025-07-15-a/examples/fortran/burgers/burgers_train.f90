program burgers_train
  use torchfort
  use, intrinsic :: iso_c_binding, only: c_float, c_int64_t
  implicit none

  integer(c_int) :: istat
  integer :: device = 0 ! 0 for GPU, -1 for CPU
  integer :: i
  real(c_float) :: loss_val
  integer, parameter :: num_epochs = 5000 ! Matching burgers1d.py

  ! Parameters from burgers1d.py
  integer, parameter :: N_f = 10000
  integer, parameter :: N_0 = 400
  integer, parameter :: N_b = 200
  real(c_float), parameter :: x_min = -1.0_c_float
  real(c_float), parameter :: x_max = 1.0_c_float
  real(c_float), parameter :: t_min = 0.0_c_float
  real(c_float), parameter :: t_max = 1.0_c_float

  ! Declare arrays with TARGET attribute for c_loc
  real(c_float), allocatable, contiguous, target :: X_f(:,:)
  real(c_float), allocatable, contiguous, target :: x0_cat(:,:)
  real(c_float), allocatable, contiguous, target :: xb_left_cat(:,:)
  real(c_float), allocatable, contiguous, target :: xb_right_cat(:,:)
  real(c_float), allocatable, contiguous, target :: u0_train(:,:)

  ! TorchFort tensor lists
  type(torchfort_tensor_list) :: input_tensors
  type(torchfort_tensor_list) :: label_tensors

  !--------------------------------------------------------------------
  ! Generate Physics-Informed Training Data (matching burgers1d.py)
  ! NOTE: Dimensions are swapped to match Python (features, N)
  !--------------------------------------------------------------------
  allocate(X_f(2, N_f))
  allocate(x0_cat(2, N_0))
  allocate(xb_left_cat(2, N_b))
  allocate(xb_right_cat(2, N_b))
  allocate(u0_train(1, N_0))

  ! 1. Collocation points (X_f)
  call random_number(X_f)
  X_f(1, :) = X_f(1, :) * (x_max - x_min) + x_min ! x in [-1, 1]
  X_f(2, :) = X_f(2, :) * (t_max - t_min) + t_min ! t in [0, 1]

  ! 2. Initial condition (t=0)
  call random_number(x0_cat)
  x0_cat(1, :) = x0_cat(1, :) * (x_max - x_min) + x_min ! x in [-1, 1]
  x0_cat(2, :) = 0.0_c_float                      ! t = 0
  u0_train(1, :) = -sin(acos(-1.0_c_float) * x0_cat(1, :))

  ! 3. Boundary conditions (x=-1 and x=1)
  call random_number(xb_left_cat)
  xb_left_cat(1, :) = x_min ! x = -1
  xb_left_cat(2, :) = xb_left_cat(2, :) * (t_max - t_min) + t_min ! t in [0, 1]
  
  call random_number(xb_right_cat)
  xb_right_cat(1, :) = x_max ! x = 1
  xb_right_cat(2, :) = xb_right_cat(2, :) * (t_max - t_min) + t_min ! t in [0, 1]

  print *, "Successfully generated physics-informed training data."
  call flush(6)

  !--------------------------------------------------------------------
  ! Initialize TorchFort and create model
  !--------------------------------------------------------------------
  istat = torchfort_create_model("burgers_model", "config_burgers_model.yaml", device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_create_model failed with status: ", istat
    stop
  endif
  print *, "Model created successfully."
  call flush(6)

  !--------------------------------------------------------------------
  ! Create and populate tensor lists for training
  !--------------------------------------------------------------------
  istat = torchfort_tensor_list_create(input_tensors)
  istat = torchfort_tensor_list_add_tensor(input_tensors, X_f)
  istat = torchfort_tensor_list_add_tensor(input_tensors, x0_cat)
  istat = torchfort_tensor_list_add_tensor(input_tensors, xb_left_cat)
  istat = torchfort_tensor_list_add_tensor(input_tensors, xb_right_cat)

  istat = torchfort_tensor_list_create(label_tensors)
  istat = torchfort_tensor_list_add_tensor(label_tensors, u0_train)

  !--------------------------------------------------------------------
  ! Training loop
  !--------------------------------------------------------------------
  print *, "Starting training loop..."
  call flush(6)

  do i = 1, num_epochs
    ! Perform a training step
    ! The first argument is the model name, second is the loss name
    istat = torchfort_train_multiarg("burgers_model", input_tensors, label_tensors, loss_val)
    if (istat /= TORCHFORT_RESULT_SUCCESS) then
        print *, "ERROR: torchfort_train failed in epoch", i, " with status: ", istat
        stop
    endif

    if (mod(i, 500) == 0) then ! Print every 500 epochs, matching burgers1d.py
      print *, "Epoch: ", i, " Loss: ", loss_val
      call flush(6)
    endif
    
  end do

  print *, "Training finished. Final loss: ", loss_val
  call flush(6)

  !--------------------------------------------------------------------
  ! Destroy tensor lists
  !--------------------------------------------------------------------
  istat = torchfort_tensor_list_destroy(input_tensors)
  istat = torchfort_tensor_list_destroy(label_tensors)

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

end program burgers_train
