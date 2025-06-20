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
  real(real32) :: X_f(2, N_f), x0_t0(2, N_0), u0(N_0, 1), xb_left_tb(2, N_b), xb_right_tb(2, N_b)

  ! 1. Read Data from text files
  ! NOTE: Fortran is column-major, while PyTorch is row-major.
  ! We read directly into the Fortran column-major format.
  open(unit=10, file="X_f.txt", status="old")
  do i = 1, N_f
    read(10, *) X_f(1, i), X_f(2, i)
  end do
  close(10)

  open(unit=11, file="x0_t0.txt", status="old")
  do i = 1, N_0
    read(11, *) x0_t0(1, i), x0_t0(2, i)
  end do
  close(11)

  open(unit=12, file="u0.txt", status="old")
  do i = 1, N_0
    read(12, *) u0(i, 1)
  end do
  close(12)

  open(unit=13, file="xb_left_tb.txt", status="old")
  do i = 1, N_b
    read(13, *) xb_left_tb(1, i), xb_left_tb(2, i)
  end do
  close(13)

  open(unit=14, file="xb_right_tb.txt", status="old")
  do i = 1, N_b
    read(14, *) xb_right_tb(1, i), xb_right_tb(2, i)
  end do
  close(14)
  
  print *, "Successfully read training data from text files."

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

    ! Add tensors to the input list for the model (BurgersPINN)
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