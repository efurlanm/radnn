program burgers03
  use, intrinsic :: iso_fortran_env, only: real32
  use torchfort
  implicit none

  integer, parameter :: num_epochs = 5000
  integer :: i, istat
  real(real32) :: loss_val
  integer :: model_device = 0
  type(torchfort_tensor_list) :: input_tensors, label_tensors, extra_args_tensors

  ! Physics and data parameters from burgers1d.py
  real(real32), parameter :: x_min = -1.0, x_max = 1.0
  real(real32), parameter :: t_min = 0.0, t_max = 1.0
  integer, parameter :: N_f = 10000, N_0 = 400, N_b = 200
  real(real32), parameter :: pi = acos(-1.0)

  ! Data arrays
  real(real32), allocatable :: X_f(:,:), x0_t0(:,:), u0(:,:), xb_left_tb(:,:), xb_right_tb(:,:)
  real(real32), allocatable :: temp_rand(:,:)

  ! 1. Allocate and Generate Data
  ! NOTE: Fortran is column-major, while PyTorch is row-major.
  ! To pass a (N, 2) tensor, we must declare it as (2, N) in Fortran.
  allocate(X_f(2, N_f))
  allocate(x0_t0(2, N_0), u0(N_0, 1))
  allocate(xb_left_tb(2, N_b), xb_right_tb(2, N_b))

  ! Generate random numbers in a temporary array with logical dimensions
  allocate(temp_rand(N_f, 2))
  call random_number(temp_rand)

  ! Assign and scale, transposing the data into the correct memory layout
  X_f(1, :) = temp_rand(:, 1) * (x_max - x_min) + x_min
  X_f(2, :) = temp_rand(:, 2) * (t_max - t_min) + t_min
  deallocate(temp_rand)

  do i = 1, N_0
    x0_t0(1, i) = x_min + (i - 1) * (x_max - x_min) / real(N_0 - 1)
    x0_t0(2, i) = 0.0
    u0(i, 1) = -sin(pi * x0_t0(1, i))
  end do

  do i = 1, N_b
    xb_left_tb(2, i) = t_min + (i - 1) * (t_max - t_min) / real(N_b - 1)
    xb_left_tb(1, i) = x_min
    xb_right_tb(2, i) = t_min + (i - 1) * (t_max - t_min) / real(N_b - 1)
    xb_right_tb(1, i) = x_max
  end do
  
  print *, "Successfully generated training data."

  ! Setup Model
  print *, "DEBUG: Before torchfort_set_cudnn_benchmark."
  call flush(6)
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to set cudnn benchmark mode"
  print *, "DEBUG: After torchfort_set_cudnn_benchmark."
  call flush(6)

  print *, "DEBUG: Before torchfort_create_model."
  call flush(6)
  istat = torchfort_create_model("mymodel", "burgers_train_config.yaml", model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to create model"
  print *, "DEBUG: After torchfort_create_model. Model created successfully."
  call flush(6)

  ! Training Loop
  print *, "DEBUG: Before training loop. Starting training for ", num_epochs, " epochs..."
  call flush(6)
  do i = 1, num_epochs
    print *, "DEBUG: In training loop, epoch: ", i
    call flush(6)

    ! Create tensor lists
    istat = torchfort_tensor_list_create(input_tensors)
    if (istat /= 0) stop "Failed to create input_tensors"
    istat = torchfort_tensor_list_create(label_tensors)
    if (istat /= 0) stop "Failed to create label_tensors"
    istat = torchfort_tensor_list_create(extra_args_tensors)
    if (istat /= 0) stop "Failed to create extra_args_tensors"

    ! Add tensors to the input list for the model
    istat = torchfort_tensor_list_add_tensor(input_tensors, X_f)
    istat = torchfort_tensor_list_add_tensor(input_tensors, x0_t0)
    istat = torchfort_tensor_list_add_tensor(input_tensors, xb_left_tb)
    istat = torchfort_tensor_list_add_tensor(input_tensors, xb_right_tb)

    ! Add tensors to the label list for the loss function
    istat = torchfort_tensor_list_add_tensor(label_tensors, u0)

    ! extra_args_tensors is empty
    
    ! Train the model
    print *, "DEBUG: Before torchfort_train_multiarg."
    call flush(6)
    istat = torchfort_train_multiarg("mymodel", input_tensors, label_tensors, loss_val, extra_args_tensors)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Training step failed"
    print *, "DEBUG: After torchfort_train_multiarg. Epoch: ", i, ", Loss: ", loss_val
    call flush(6)

    if (mod(i, 100) == 0) then
      print *, "Epoch: ", i, ", Loss: ", loss_val
    end if

    ! Destroy tensor lists
    istat = torchfort_tensor_list_destroy(input_tensors)
    istat = torchfort_tensor_list_destroy(label_tensors)
    istat = torchfort_tensor_list_destroy(extra_args_tensors)
  end do
  print *, "DEBUG: After training loop. Training complete!"
  call flush(6)

  ! 4. Save Model
  istat = torchfort_save_model("mymodel", "burgers_model_trained.pt")
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to save model"
  print *, "Model saved to burgers_model_trained.pt"

end program burgers03