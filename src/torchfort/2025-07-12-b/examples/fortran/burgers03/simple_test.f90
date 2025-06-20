program simple_test
  use, intrinsic :: iso_fortran_env, only: real32
  use torchfort
  implicit none

  integer :: i, istat
  real(real32) :: loss_val
  integer :: model_device = 0
  type(torchfort_tensor_list) :: input_tensors, label_tensors, extra_args_tensors
  real(real32), allocatable :: tensor1(:), tensor2(:,:), tensor3(:), tensor4(:,:), tensor5(:,:), tensor6(:)

  ! Allocate dummy tensors
  allocate(tensor1(10))
  allocate(tensor2(2, 20))
  allocate(tensor3(5))
  allocate(tensor4(2, 15))
  allocate(tensor5(2, 15))
  allocate(tensor6(1))
  call random_number(tensor1)
  call random_number(tensor2)
  call random_number(tensor3)
  call random_number(tensor4)
  call random_number(tensor5)
  call random_number(tensor6)

  ! Setup Model
  istat = torchfort_create_model("simple_model", "simple_config.yaml", model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop "FATAL: Failed to create simple_model"
  print *, "Simple model created."

  ! Create tensor lists
  istat = torchfort_tensor_list_create(input_tensors)
  istat = torchfort_tensor_list_create(label_tensors)
  istat = torchfort_tensor_list_create(extra_args_tensors)

  ! Add 6 tensors to the input list
  print *, "Adding 6 tensors to the input list..."
  istat = torchfort_tensor_list_add_tensor(input_tensors, tensor1)
  istat = torchfort_tensor_list_add_tensor(input_tensors, tensor2)
  istat = torchfort_tensor_list_add_tensor(input_tensors, tensor3)
  istat = torchfort_tensor_list_add_tensor(input_tensors, tensor4)
  istat = torchfort_tensor_list_add_tensor(input_tensors, tensor5)
  istat = torchfort_tensor_list_add_tensor(input_tensors, tensor6)

  ! Train the model
  print *, "Calling torchfort_train_multiarg..."
  istat = torchfort_train_multiarg("simple_model", input_tensors, label_tensors, loss_val, extra_args_tensors)
  if (istat /= TORCHFORT_RESULT_SUCCESS) then
      print *, "Training step failed as expected for this test."
  else
      print *, "Training step succeeded. Loss: ", loss_val
  end if

  ! Destroy tensor lists
  istat = torchfort_tensor_list_destroy(input_tensors)
  istat = torchfort_tensor_list_destroy(label_tensors)
  istat = torchfort_tensor_list_destroy(extra_args_tensors)

end program simple_test
