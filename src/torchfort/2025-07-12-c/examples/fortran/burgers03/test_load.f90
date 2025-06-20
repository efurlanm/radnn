program test_load
  use, intrinsic :: iso_fortran_env, only: real32
  use torchfort
  implicit none

  integer :: istat

  print *, "Attempting to load simple_model.pt..."

  istat = torchfort_create_model("simple_model", "simple_model.pt", -1)

  if (istat == TORCHFORT_RESULT_SUCCESS) then
    print *, "SUCCESS: Model loaded without error."
  else
    print *, "FAILURE: Model loading failed with status: ", istat
  end if

end program test_load
