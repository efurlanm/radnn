program burgers04
  use torchfort
  implicit none

  integer :: istat, i, j, epoch
  character(len=256) :: configfile = "../examples/fortran/burgers/config_burgers.yaml"
  character(len=20) :: model_name = "burgers_model"
  character(len=256) :: trained_model_fname = "../examples/fortran/burgers/burgers_model_trained_04.pt"

  real(real32), allocatable, target :: t_collocation(:,:), x_collocation(:,:)
  real(real32), allocatable, target :: t_initial(:,:), x_initial(:,:), u_initial_true(:,:)
  real(real32), allocatable, target :: t_boundary(:,:), x_boundary(:,:), u_boundary_true(:,:)
  real(real32) :: nu
  integer, parameter :: N_f = 10000, N_0 = 400, N_b = 200 ! Updated N_0
  real(real32), parameter :: x_min = -1.0, x_max = 1.0, t_min = 0.0, t_max = 1.0
  integer, parameter :: num_epochs = 50000 ! Updated num_epochs

  type(torchfort_tensor_list) :: inputs_list, labels_list
  real(real32) :: loss_val
  real(real32), allocatable :: dummy_label_tensor(:)

  ! Fixed random seed for reproducibility
  integer, parameter :: SEED = 42

  ! Define physical parameters
  nu = 0.01 / 3.14159265358979323846_real32 ! pi

  ! Set random seed for Fortran's random_number intrinsic
  call srand(SEED)

  ! setup the training model
  istat = torchfort_create_model(model_name, configfile, TORCHFORT_DEVICE_CPU)
  if (istat /= 0) then
     print *, "Error creating training model"
     stop
  endif
  print *, "Training model created successfully"

  ! Generate collocation points for PDE
  allocate(t_collocation(1, N_f))
  allocate(x_collocation(1, N_f))
  call random_number(t_collocation)
  call random_number(x_collocation)
  x_collocation = x_collocation * (x_max - x_min) + x_min
  t_collocation = t_collocation * (t_max - t_min) + t_min

  ! Initial condition u(x, 0) = -sin(pi * x)
  allocate(x_initial(1, N_0))
  allocate(t_initial(1, N_0))
  allocate(u_initial_true(1, N_0))
  do i = 1, N_0
     x_initial(1, i) = x_min + (i-1) * (x_max - x_min) / (N_0 - 1)
     t_initial(1, i) = t_min
     u_initial_true(1, i) = -sin(3.14159265358979323846_real32 * x_initial(1, i))
  end do

  ! Boundary conditions u(-1,t) = 0 and u(1,t) = 0
  allocate(t_boundary(1, N_b))
  allocate(x_boundary(1, N_b))
  allocate(u_boundary_true(1, N_b))
  do i = 1, N_b
     t_boundary(1, i) = t_min + (i-1) * (t_max - t_min) / (N_b - 1)
     x_boundary(1, i) = x_min ! Left boundary
     u_boundary_true(1, i) = 0.0_real32
  end do

  ! Create tensor lists for training
  istat = torchfort_tensor_list_create(inputs_list)
  if (istat /= 0) then
     print *, "Error creating inputs_list"
     stop
  endif
  istat = torchfort_tensor_list_create(labels_list)
  if (istat /= 0) then
     print *, "Error creating labels_list"
     stop
  endif

  ! Add all inputs required by BurgersPINN's forward method
  istat = torchfort_tensor_list_add_tensor(inputs_list, t_collocation)
  istat = torchfort_tensor_list_add_tensor(inputs_list, x_collocation)
  istat = torchfort_tensor_list_add_tensor(inputs_list, t_initial)
  istat = torchfort_tensor_list_add_tensor(inputs_list, x_initial)
  istat = torchfort_tensor_list_add_tensor(inputs_list, u_initial_true)
  istat = torchfort_tensor_list_add_tensor(inputs_list, t_boundary)
  istat = torchfort_tensor_list_add_tensor(inputs_list, x_boundary)
  istat = torchfort_tensor_list_add_tensor(inputs_list, u_boundary_true)

  ! The labels_list will receive the loss value from the model.
  ! We need a dummy tensor in labels_list for the IdentityLoss to work.
  ! The actual value of this tensor doesn't matter, only its presence.
  ! We will allocate it as a scalar array (1 element).
  allocate(dummy_label_tensor(1))
  dummy_label_tensor = 0.0_real32 ! Initialize with a dummy value
  istat = torchfort_tensor_list_add_tensor(labels_list, dummy_label_tensor)
  if (istat /= 0) then
     print *, "Error adding dummy_label_tensor to labels_list"
     stop
  endif

  ! Training loop
  do epoch = 1, num_epochs
     istat = torchfort_train_multiarg(model_name, inputs_list, labels_list, loss_val)
     if (istat /= 0) then
        print *, "Error during training at epoch ", epoch
        stop
     endif
     ! Print loss every 5000 epochs to track progress (updated from 100)
     if (mod(epoch, 5000) == 0) then
        print *, "Epoch: ", epoch, ", Loss: ", loss_val
     endif
  end do
  print *, "Training completed. Final Loss: ", loss_val

  ! Save the trained model
  istat = torchfort_save_model(model_name, trained_model_fname)
  if (istat /= 0) then
     print *, "Error saving trained model"
     stop
  endif
  print *, "Trained model saved successfully to ", trained_model_fname

  ! Deallocate tensor lists and tensors
  istat = torchfort_tensor_list_destroy(inputs_list)
  istat = torchfort_tensor_list_destroy(labels_list)
  deallocate(t_collocation)
  deallocate(x_collocation)
  deallocate(t_initial)
  deallocate(x_initial)
  deallocate(u_initial_true)
  deallocate(t_boundary)
  deallocate(x_boundary)
  deallocate(u_boundary_true)
  deallocate(dummy_label_tensor)

end program burgers04