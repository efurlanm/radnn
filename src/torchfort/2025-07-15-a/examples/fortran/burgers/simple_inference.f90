program simple_inference
  use iso_c_binding, only: c_float, c_int
  use torchfort

  implicit none

  ! Model and tensor handles
  type(torchfort_model_handle) :: model_handle
  type(torchfort_tensor_list) :: input_tensors, output_tensors

  ! Input and output data
  real(c_float), allocatable, contiguous, target :: input_data(:,:)
  real(c_float), allocatable, contiguous, target :: output_data(:,:)

  ! Status variable
  integer :: istat

  ! Configuration file path
  character(len=*), parameter :: config_file = "config_simple_inference.yaml"

  !--------------------------------------------------------------------
  ! Initialize TorchFort
  !--------------------------------------------------------------------
  istat = torchfort_init()
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_init failed with status: ", istat
    stop
  end if

  !--------------------------------------------------------------------
  ! Create model from configuration
  !--------------------------------------------------------------------
  print *, "Creating model from configuration: ", config_file
  call flush(6)
  istat = torchfort_create_model(model_handle, config_file)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_create_model failed with status: ", istat
    stop
  end if
  print *, "Model created successfully."
  call flush(6)

  !--------------------------------------------------------------------
  ! Prepare input data
  !--------------------------------------------------------------------
  ! Allocate input_data as (N, features) to match PyTorch's row-major expectation
  ! For a single sample with 2 features, this is (1, 2)
  allocate(input_data(1, 2))
  input_data(1, 1) = 0.5_c_float
  input_data(1, 2) = 0.5_c_float

  ! Allocate output_data
  allocate(output_data(1, 1)) ! Output is (N, output_features) = (1, 1)

  ! Create and populate tensor lists for inference
  istat = torchfort_tensor_list_create(input_tensors)
  istat = torchfort_tensor_list_add_tensor(input_tensors, input_data)

  istat = torchfort_tensor_list_create(output_tensors)
  istat = torchfort_tensor_list_add_tensor(output_tensors, output_data)

  print *, "Starting inference..."
  call flush(6)

  !--------------------------------------------------------------------
  ! Perform inference
  !--------------------------------------------------------------------
  istat = torchfort_inference(model_handle, input_tensors, output_tensors)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_inference failed with status: ", istat
    stop
  end if

  print *, "Inference complete."
  call flush(6)

  !--------------------------------------------------------------------
  ! Print results
  !--------------------------------------------------------------------
  print *, "Fortran Input: ", input_data(1,1), input_data(1,2)
  print *, "Fortran Output: ", output_data(1,1)
  call flush(6)

  !--------------------------------------------------------------------
  ! Finalize TorchFort
  !--------------------------------------------------------------------
  istat = torchfort_finalize()
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "ERROR: torchfort_finalize failed with status: ", istat
    stop
  end if

end program simple_inference
