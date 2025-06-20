program burgers03
  use iso_c_binding
  implicit none

  ! Training parameters
  integer, parameter :: num_epochs = 5000
  real(c_float), parameter :: learning_rate = 1e-3

  ! Data dimensions
  integer, parameter :: N_f = 10000
  integer, parameter :: N_0 = 400
  integer, parameter :: N_b = 200
  integer, parameter :: num_points_x = 256
  integer, parameter :: num_points_t = 100

  ! Model names and paths
  character(len=*), parameter :: model_name = "burgers_model"
  character(len=*), parameter :: configfile = "config_burgers.yaml"
  character(len=*), parameter :: trained_model_fname = "burgers_model_trained.pt"
  character(len=*), parameter :: python_original_results_fname = "burgers1d_python_original_results.bin"
  character(len=*), parameter :: fortran_inference_output_fname = "burgers_fortran_inference_results.bin"

  ! Variables
  type(c_ptr) :: optimizer
  real(c_float) :: loss_val

  real(c_float), allocatable :: x_original_python(:)
  real(c_float), allocatable :: t_original_python(:)
  real(c_float), allocatable :: u_pred_original_python(:)

  real(c_float), allocatable :: input_train_data(:,:)
  real(c_float), allocatable :: u_label_fortran(:,:)

  real(c_float), allocatable :: input_inference_data(:,:)
  real(c_float), allocatable :: u_inference_fortran(:,:)

  integer :: i, j, k, istat
  integer :: read_num_x, read_num_t

  ! Interfaces for TorchFort functions
  interface
    function torchfort_initialize() bind(C, name="torchfort_initialize") result(res)
      import
      integer(c_int) :: res
    end function torchfort_initialize

    function torchfort_set_manual_seed(seed) bind(C, name="torchfort_set_manual_seed") result(res)
      import
      integer(c_int) :: seed
      integer(c_int) :: res
    end function torchfort_set_manual_seed

    function torchfort_create_model(mname, fname, dev) bind(C, name="torchfort_create_model") result(res)
      import
      character(kind=c_char) :: mname(*), fname(*)
      integer(c_int), value :: dev
      integer(c_int) :: res
    end function torchfort_create_model

    subroutine torchfort_optim_adam(mname, lr, optimizer) bind(C, name="torchfort_optim_adam")
      import
      character(kind=c_char) :: mname(*)
      real(c_float), value :: lr
      type(c_ptr), value :: optimizer
    end subroutine torchfort_optim_adam

    subroutine torchfort_optim_zero_grad(optimizer) bind(C, name="torchfort_optim_zero_grad")
      import
      type(c_ptr), value :: optimizer
    end subroutine torchfort_optim_zero_grad

    function torchfort_train_float_2d(mname, input, input_dim1, input_dim2, label, label_dim1, label_dim2, loss_val, dtype, stream) bind(C, name="torchfort_train_F") result(res)
      import
      character(kind=c_char) :: mname(*)
      real(c_float) :: input(input_dim1, input_dim2), label(label_dim1, label_dim2)
      real(c_float) :: loss_val
      integer(c_int64_t), value :: input_dim1, input_dim2, label_dim1, label_dim2
      integer(c_int), value :: dtype
      integer(c_int64_t), value :: stream
      integer(c_int) :: res
    end function torchfort_train_float_2d

    subroutine torchfort_optim_step(optimizer) bind(C, name="torchfort_optim_step")
      import
      type(c_ptr), value :: optimizer
    end subroutine torchfort_optim_step

    function torchfort_save_model(mname, fname) bind(C, name="torchfort_save_model") result(res)
      import
      character(kind=c_char) :: mname(*), fname(*)
      integer(c_int) :: res
    end function torchfort_save_model

    function torchfort_inference_float_2d(mname, input, input_dim1, input_dim2, output, output_dim1, output_dim2, dtype, stream) bind(C, name="torchfort_inference_F") result(res)
      import
      character(kind=c_char) :: mname(*)
      real(c_float) :: input(input_dim1, input_dim2), output(output_dim1, output_dim2)
      integer(c_int64_t), value :: input_dim1, input_dim2, output_dim1, output_dim2
      integer(c_int), value :: dtype
      integer(c_int64_t), value :: stream
      integer(c_int) :: res
    end function torchfort_inference_float_2d

    function torchfort_finalize() bind(C, name="torchfort_finalize") result(res)
      import
      integer(c_int) :: res
    end function torchfort_finalize

  end interface

  ! TorchFort constants (from torchfort_enums.h)
  integer, parameter :: TORCHFORT_DEVICE_CPU = -1
  integer, parameter :: TORCHFORT_RESULT_SUCCESS = 0

  istat = torchfort_initialize()
  istat = torchfort_set_manual_seed(42)

  ! Setup the training model (using config file)
  istat = torchfort_create_model(model_name // C_NULL_CHAR, configfile // C_NULL_CHAR, TORCHFORT_DEVICE_CPU)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
     print *, "Error creating training model"
     stop
  endif
  print *, "Training model created successfully"

  ! --- Read original Python results for training labels ---
  open(unit=10, file=python_original_results_fname, form='unformatted', access='stream', status='old')
  read(10) read_num_x, read_num_t

  allocate(x_original_python(read_num_x))
  allocate(t_original_python(read_num_t))
  allocate(u_pred_original_python(read_num_x * read_num_t))

  read(10) x_original_python
  read(10) t_original_python
  read(10) u_pred_original_python
  close(10)
  print *, "Original Python results loaded for training labels."

  ! Prepare training data for Fortran model
  ! The Fortran model will learn to map (x,t) to u_pred from Python
  allocate(input_train_data(2, read_num_x * read_num_t))
  allocate(u_label_fortran(1, read_num_x * read_num_t))

  ! Flatten X and T meshgrid from Python for Fortran input
  k = 1
  do j = 1, read_num_t
    do i = 1, read_num_x
      input_train_data(1, k) = x_original_python(i)
      input_train_data(2, k) = t_original_python(j)
      u_label_fortran(1, k) = u_pred_original_python(k)
      k = k + 1
    end do
  end do

  ! Create optimizer
  call torchfort_optim_adam(model_name // C_NULL_CHAR, learning_rate, optimizer)

  ! Training loop
  print *, "Starting training..."
  do i = 1, num_epochs
    call torchfort_optim_zero_grad(optimizer)
    istat = torchfort_train_float_2d(model_name // C_NULL_CHAR, input_train_data, size(input_train_data, 1, kind=c_int64_t), size(input_train_data, 2, kind=c_int64_t), &
                                     u_label_fortran, size(u_label_fortran, 1, kind=c_int64_t), size(u_label_fortran, 2, kind=c_int64_t), &
                                     loss_val, 0, 0_c_int64_t) ! Use specific 2D float train
    if (istat /= TORCHFORT_RESULT_SUCCESS) then
       print *, "Error during training at epoch ", i
       stop
    endif
    call torchfort_optim_step(optimizer)

    if (mod(i, 100) == 0) then
      print *, "Epoch:", i, "Loss:", loss_val
    end if

  end do
  print *, "Training finished."

  ! Save the trained model
  istat = torchfort_save_model(model_name // C_NULL_CHAR, trained_model_fname // C_NULL_CHAR)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
     print *, "Error saving trained model"
     stop
  endif
  print *, "Trained model saved to ", trained_model_fname

  ! --- Perform Inference on a grid and save results ---

  ! Prepare inference grid (same as original Python for comparison)
  allocate(input_inference_data(2, num_points_x * num_points_t))
  allocate(u_inference_fortran(1, num_points_x * num_points_t))

  k = 1
  do j = 1, num_points_t
    do i = 1, num_points_x
      input_inference_data(1, k) = x_original_python(i)
      input_inference_data(2, k) = t_original_python(j)
      k = k + 1
    end do
  end do

  ! Perform inference using the trained model
  istat = torchfort_inference_float_2d(model_name // C_NULL_CHAR, input_inference_data, size(input_inference_data, 1, kind=c_int64_t), size(input_inference_data, 2, kind=c_int64_t), &
                                       u_inference_fortran, size(u_inference_fortran, 1, kind=c_int64_t), size(u_inference_fortran, 2, kind=c_int64_t), &
                                       0, 0_c_int64_t) ! Use specific 2D float inference
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
     print *, "Error during inference"
     stop
  endif
  print *, "Inference completed successfully."

  ! Save inference results to binary file
  open(unit=10, file=fortran_inference_output_fname, form='unformatted', access='stream', status='replace')
  write(10) num_points_x, num_points_t
  write(10) x_original_python
  write(10) t_original_python
  write(10) u_inference_fortran
  close(10)
  print *, "Inference results saved to ", fortran_inference_output_fname

  ! Finalize
  istat = torchfort_finalize()

  ! Deallocate tensors
  deallocate(x_original_python)
  deallocate(t_original_python)
  deallocate(u_pred_original_python)
  deallocate(input_train_data)
  deallocate(u_label_fortran)
  deallocate(input_inference_data)
  deallocate(u_inference_fortran)

contains
  ! Helper function to get shape array for torchfort_tensor_list_add_tensor
  function shape(arr) result(res)
    integer, dimension(:), intent(in) :: arr
    integer(c_int64_t), dimension(size(arr)) :: res
    res = arr
  end function shape

end program burgers03
