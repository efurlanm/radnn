! sum_mpi.f90
PROGRAM DistributedSum

  IMPLICIT NONE

  INCLUDE 'mpif.h' ! Inclui as definições do MPI para Fortran

  INTEGER :: rank, size, ierror
  INTEGER :: local_sum, global_sum
  INTEGER :: i

  ! Inicializa o ambiente MPI
  CALL MPI_INIT(ierror)

  ! Obtém o rank (ID) do processo atual
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

  ! Obtém o número total de processos
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)

  ! Cada processo calcula uma soma local baseada no seu rank
  local_sum = 0
  DO i = 1, rank + 10 ! Apenas para ter uma soma diferente por rank
    local_sum = local_sum + i
  END DO

  PRINT *, 'Process ', rank, ' (of ', size, ') has local_sum = ', local_sum

  ! Soma todos os 'local_sum' de todos os processos para obter 'global_sum'
  ! MPI_REDUCE opera em todos os processos em um comunicador.
  ! MPI_SUM é a operação de soma.
  ! 0 é o rank do processo raiz que receberá o resultado final.
  CALL MPI_REDUCE(local_sum, global_sum, 1, MPI_INTEGER, MPI_SUM, 0, &
       MPI_COMM_WORLD, ierror)

  ! Apenas o processo com rank 0 (raiz) imprime o resultado final
  IF (rank .EQ. 0) THEN
    PRINT *, '----------------------------------'
    PRINT *, 'Global sum calculated by root (rank 0): ', global_sum
    PRINT *, '----------------------------------'
  END IF

  ! Finaliza o ambiente MPI
  CALL MPI_FINALIZE(ierror)

END PROGRAM DistributedSum 
