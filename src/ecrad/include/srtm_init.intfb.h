INTERFACE
SUBROUTINE SRTM_INIT(DIRECTORY, NWVCONTINUUM)
USE PARKIND1, ONLY : JPIM
CHARACTER(LEN=*), INTENT(IN) :: DIRECTORY
INTEGER(KIND=JPIM), INTENT(IN), OPTIONAL :: NWVCONTINUUM
END SUBROUTINE SRTM_INIT
END INTERFACE