program test_torchfort
  use torchfort
  implicit none

  integer :: istat

  print *, "Attempting to create model..."
  call flush(6)
  
  istat = torchfort_create_model("test_model", "config_mlp_native.yaml", -1)
  
  print *, "torchfort_create_model returned with status: ", istat
  call flush(6)

  if (istat /= TORCHFORT_RESULT_SUCCESS) then
    print *, "FATAL: Failed to create model. Stopping."
    call flush(6)
    stop 1
  end if

  print *, "Model created successfully."
  call flush(6)

end program test_torchfort