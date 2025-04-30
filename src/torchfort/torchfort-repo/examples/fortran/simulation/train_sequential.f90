! SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

subroutine print_help_message
  print*, &
  "Usage: train_sequential [options]\n"// & ! Nome do programa alterado
  "options:\n"// &
  "\t--configfile\n" // &
  "\t\tTorchFort configuration file to use. (default: config_mlp_native.yaml) \n" // &
  "\t--simulation_device\n" // &
  "\t\tDevice to run simulation on. (-1 for CPU, 0 for GPU. default: GPU) \n" // &
  "\t--train_device\n" // &
  "\t\tDevice to run model training/inference on. (-1 for CPU, 0 for GPU. default: GPU) \n" // &
  "\t--ntrain_steps\n" // &
  "\t\tNumber of training steps to run. (default: 100000) \n" // &
  "\t--nval_steps\n" // &
  "\t\tNumber of validation steps to run. (default: 1000) \n" // &
  "\t--val_write_freq\n" // &
  "\t\tFrequency to write validation HDF5 files. (default: 10) \n" // &
  "\t--checkpoint_dir\n" // &
  "\t\tCheckpoint directory to load. (default: don't load checkpoint) \n" // &
  "\t--output_model_name\n" // &
  "\t\tFilename for saved model. (default: model.pt) \n" // &
  "\t--output_checkpoint_dir\n" // &
  "\t\tName of checkpoint directory to save. (default: checkpoint) \n"
end subroutine print_help_message

! Programa principal, renomeado para refletir a execução sequencial
program train_sequential
  use, intrinsic :: iso_fortran_env, only: real32, real64
#ifdef _OPENACC
  use openacc
#endif
  ! use mpi ! <<< Removido

  use simulation ! Presume que 'simulation' pode rodar no grid completo
  use torchfort ! Presume que 'torchfort' tem interface não-distribuída

  implicit none

  integer :: i, j, istat
  integer :: n, nchannels, batch_size
  real(real32) :: a(2), dt
  real(real32) :: loss_val
  real(real64) :: mse
  ! Arrays agora para o tamanho GLOBAL, não local por rank
  real(real32), allocatable :: u(:,:), u_div(:,:)
  ! Arrays de dados de treinamento/inferência para o batch size total
  real(real32), allocatable :: input(:,:,:,:), label(:,:,:,:), output(:,:,:,:)

  character(len=7) :: idx
  character(len=256) :: filename
  logical :: load_ckpt = .false.
  integer :: train_step_ckpt = 0
  integer :: val_step_ckpt = 0

  ! Simula um único rank (0 de 1)
  integer, parameter :: rank = 0
  integer, parameter :: local_rank = 0
  integer, parameter :: nranks = 1 ! <<< Definido como 1

#ifdef _OPENACC
  integer(acc_device_kind) :: dev_type
#endif
  ! integer, allocatable :: sendcounts(:), recvcounts(:) ! <<< Removido
  ! integer, allocatable :: sdispls(:), rdispls(:) ! <<< Removido

  ! command line arguments
  character(len=256) :: configfile = "config_mlp_native.yaml"
  character(len=256) :: checkpoint_dir
  character(len=256) :: output_model_name = "model.pt"
  character(len=256) :: output_checkpoint_dir = "checkpoint"
  integer :: ntrain_steps = 100000
  integer :: nval_steps = 1000
  integer :: val_write_freq = 10
  integer :: model_device = 0 ! <<< Default para GPU 0
  integer :: simulation_device = 0 ! <<< Default para GPU 0

  logical :: skip_next
  character(len=256) :: arg

  ! initialize MPI - REMOVIDO
  ! call MPI_Init(istat)
  ! call MPI_Comm_rank(MPI_COMM_WORLD, rank, istat)
  ! call MPI_Comm_size(MPI_COMM_WORLD, nranks, istat)
  ! call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, istat)
  ! call MPI_Comm_rank(local_comm, local_rank, istat)

  ! Verificação de nranks removida
  ! if (nranks /= 2) then
  !   print*, "This example requires 2 ranks to run. Exiting."
  !   stop
  ! endif

  ! read command line arguments
  skip_next = .false.
  do i = 1, command_argument_count()
    if (skip_next) then
      skip_next = .false.
      cycle
    end if
    call get_command_argument(i, arg)
    select case(arg)
      case('--configfile')
        call get_command_argument(i+1, arg)
        read(arg, *) configfile
        skip_next = .true.
      case('--checkpoint_dir')
        call get_command_argument(i+1, arg)
        read(arg, *) checkpoint_dir
        skip_next = .true.
        load_ckpt = .true.
      case('--output_model_name')
        call get_command_argument(i+1, arg)
        read(arg, *) output_model_name
        skip_next = .true.
      case('--output_checkpoint_dir')
        call get_command_argument(i+1, arg)
        read(arg, *) output_checkpoint_dir
        skip_next = .true.
      case('--ntrain_steps')
        call get_command_argument(i+1, arg)
        read(arg, *) ntrain_steps
        skip_next = .true.
      case('--nval_steps')
        call get_command_argument(i+1, arg)
        read(arg, *) nval_steps
        skip_next = .true.
      case('--val_write_freq')
        call get_command_argument(i+1, arg)
        read(arg, *) val_write_freq
        skip_next = .true.
      case('--train_device')
        call get_command_argument(i+1, arg)
        read(arg, *) model_device
        ! Check valid device index (0 for GPU, -1 for CPU)
        if (model_device /= -1 .and. model_device /= 0) then
          print*, "Invalid train device type argument. Use -1 for CPU or 0 for GPU."
          call exit(1)
        endif
        skip_next = .true.
      case('--simulation_device')
        call get_command_argument(i+1, arg)
        read(arg, *) simulation_device
        ! Check valid device index (0 for GPU, -1 for CPU)
        if (simulation_device /= -1 .and. simulation_device /= 0) then
          print*, "Invalid simulation device type argument. Use -1 for CPU or 0 for GPU."
          call exit(1)
        endif
        skip_next = .true.
      case('-h')
        ! if (rank == 0) then ! Não precisa mais checar rank
          call print_help_message
        ! endif
        ! call MPI_Finalize(istat) ! REMOVIDO
        call exit(0)
      case default
        print*, "Unknown argument."
        call exit(1)
    end select
  end do

#ifndef _OPENACC
  if (simulation_device /= -1) then
    print*, "OpenACC support required to run simulation on GPU. &
             Set --simulation_device -1 to run simulation on CPU."
    call exit(1)
  endif
#endif
#ifdef _OPENACC
  if (simulation_device == 0) then
    ! assign GPUs by local rank - agora usa o device especificado (0 para GPU)
    dev_type = acc_get_device_type()
    ! Usa o valor de simulation_device para selecionar a GPU (0)
    call acc_set_device_num(simulation_device, dev_type)
    call acc_init(dev_type)
  endif
#endif
  ! Assign GPU for the model - agora usa o device especificado (0 para GPU)
  ! model_device = local_rank ! <<< Removido

  ! Impressão de settings (mantida para rank 0, que agora é o único)
  if (rank == 0) then
    print*, "Run settings:"
    print*, "\tconfigfile: ", trim(configfile)
    if (simulation_device == TORCHFORT_DEVICE_CPU) then
      print*, "\tsimulation_device: cpu"
    else
      print*, "\tsimulation_device: gpu (device ", simulation_device, ")"
    endif
    if (model_device == TORCHFORT_DEVICE_CPU) then
      print*, "\ttrain_device: cpu"
    else
      print*, "\ttrain_device: gpu (device ", model_device, ")"
    endif
    if (load_ckpt) then
      print*, "\tcheckpoint_dir: ", trim(checkpoint_dir)
    else
      print*, "\tcheckpoint_dir:", "NONE"
    endif
    print*, "\toutput_model_name: ", trim(output_model_name)
    print*, "\toutput_checkpoint_dir: ", trim(output_checkpoint_dir)
    print*, "\tntrain_steps: ", ntrain_steps
    print*, "\tnval_steps: ", nval_steps
    print*, "\tval_write_freq: ", val_write_freq
    print*
  endif

  ! model/simulation parameters
  n = 32
  nchannels = 1
  ! batch_size agora é o batch size total, não por rank
  batch_size = 16 ! <<< batch_size = 16 / nranks, com nranks=1, fica 16
  dt = 0.01
  a = [1.0, 0.789] ! off-angle to generate more varied training data

  ! allocate "simulation" data sized for *GLOBAL* domain
  allocate(u(n, n)) ! <<< Tamanho total
  allocate(u_div(n, n)) ! <<< Tamanho total

  ! allocate training/inference data in standard 2D layout (NCHW, row-major),
  ! sized for *GLOBAL* domain and *TOTAL* batch size
  ! input_local e label_local não são mais necessários
  ! allocate(input_local(n, n/nranks, nchannels, batch_size*nranks)) ! <<< Removido
  ! allocate(label_local(n, n/nranks, nchannels, batch_size*nranks)) ! <<< Removido
  allocate(input(n, n, nchannels, batch_size)) ! <<< Tamanho total e batch total
  allocate(label(n, n, nchannels, batch_size)) ! <<< Tamanho total e batch total
  ! Output para inferência. O exemplo original usava batch=1 para inf,
  ! mas alocamos para o batch total por conveniência. O slicing controla o uso.
  allocate(output(n, n, nchannels, batch_size))

  ! allocate and set up arrays for MPI Alltoallv (batch redistribution) - REMOVIDO
  ! allocate(sendcounts(nranks), recvcounts(nranks))
  ! allocate(sdispls(nranks), rdispls(nranks))
  ! do i = 1, nranks
  !   sendcounts(i) = n * n/nranks
  !   recvcounts(i) = n * n/nranks
  ! end do
  ! sdispls(1) = 0
  ! rdispls(1) = 0
  ! do i = 2, nranks
  !   sdispls(i) = sdispls(i-1) + n*n/nranks*batch_size
  !   rdispls(i) = rdispls(i-1) + n*n/nranks
  ! end do

  ! set torch benchmark mode
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! setup the data parallel model - USAR VERSÃO NÃO DISTRIBUÍDA
  ! istat = torchfort_create_distributed_model("mymodel", configfile, MPI_COMM_WORLD, model_device)
  istat = torchfort_create_model("mymodel", configfile, model_device) ! <<< Usando versão não-distribuída
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! load training checkpoint if requested
  if (load_ckpt) then
    if (rank == 0) print*, "loading checkpoint..."
    ! A função de checkpoint provavelmente funciona para o modelo singular
    istat = torchfort_load_checkpoint("mymodel", checkpoint_dir, train_step_ckpt, val_step_ckpt)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  endif

  ! Inicializa a simulação - presume que init_simulation lida com o grid total
  call init_simulation(n, dt, a, train_step_ckpt*batch_size*dt, rank, nranks, simulation_device)

  ! run training
  if (rank == 0 .and. ntrain_steps >= 1) print*, "start training..."
  ! OpenACC data pragma ajustado para os novos tamanhos globais dos arrays
  ! Removido input_local e label_local
  !$acc data copyin(u, u_div, input, label) copyout(output) if(simulation_device >= 0)
  do i = 1, ntrain_steps
    ! Gerar um batch total de dados (batch_size = 16)
    do j = 1, batch_size ! Loop sobre o batch size total
      call run_simulation_step(u, u_div) ! Presume que run_simulation_step atualiza u e u_div globalmente
      ! Copiar do grid da simulação (u, u_div) para o tensor de treinamento (input, label) para este item do batch (j)
      ! Ajustado para copiar o grid completo (n, n)
      !$acc kernels if(simulation_device >= 0) async
      input(:,:,1,j) = u
      label(:,:,1,j) = u_div
      !$acc end kernels
    end do

    !$acc wait ! Espera as cópias OpenACC assíncronas, se aplicável

    ! distribute local batch data across GPUs for data parallel training - REMOVIDO
    ! Não há mais distribuição, os dados já estão no array global 'input'/'label'
    ! do j = 1, batch_size
    !   !$acc host_data use_device(input_local, label_local, input, label) if(simulation_device >= 0)
    !   call MPI_Alltoallv(input_local(:,:,1,j), sendcounts, sdispls, MPI_FLOAT, &
    !                      input(:,:,1,j), recvcounts, rdispls, MPI_FLOAT, &
    !                      MPI_COMM_WORLD, istat)
    !   call MPI_Alltoallv(label_local(:,:,1,j), sendcounts, sdispls, MPI_FLOAT, &
    !                      label(:,:,1,j), recvcounts, rdispls, MPI_FLOAT, &
    !                      MPI_COMM_WORLD, istat)
    !   !$acc end host_data
    ! end do

    !$acc wait ! Espera pelas comunicações MPI (REMOVIDO, não precisa mais)

    ! Treinar o modelo com o batch total.
    ! input e label agora são arrays globais com batch_size total
    !$acc host_data use_device(input, label) if(simulation_device >= 0)
    istat = torchfort_train("mymodel", input, label, loss_val) ! Presume que train lida com batch size total
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    !$acc end host_data
    !$acc wait ! Espera o treinamento terminar, se assíncrono
  end do
  !$acc end data ! Fim do data region para treinamento
  if (rank == 0 .and. ntrain_steps >= 1) print*, "final training loss: ", loss_val

  ! run inference
  if (rank == 0 .and. nval_steps >= 1) print*, "start validation..."
  ! Nova data region para inferência. output agora é copyout.
  ! Ajustado para novos tamanhos globais
  ! Removido input_local e label_local
  !$acc data copyin(u, u_div, input, label) copyout(output) if(simulation_device >= 0)
  do i = 1, nval_steps
    ! Gerar uma única amostra para validação (o loop original também gerava apenas 1 por passo de val)
    call run_simulation_step(u, u_div) ! Presume que run_simulation_step atualiza u e u_div globalmente

    ! Copiar para o tensor de inferência (batch size 1)
    ! Ajustado para copiar o grid completo (n, n)
    !$acc kernels async if(simulation_device >= 0)
    input(:,:,1,1) = u
    label(:,:,1,1) = u_div
    !$acc end kernels

    !$acc wait ! Espera as cópias OpenACC assíncronas, se aplicável

    ! gather sample on all GPUs - REMOVIDO
    ! Não há mais gather, a amostra de simulação já está no array global 'input'/'label'
    ! !$acc host_data use_device(input_local, label_local, input, label) if(simulation_device >= 0)
    ! call MPI_Allgather(input_local(:,:,1,1), n * n/nranks, MPI_FLOAT, &
    !                      input(:,:,1,1), n * n/nranks, MPI_FLOAT, &
    !                      MPI_COMM_WORLD, istat)
    ! call MPI_Allgather(label_local(:,:,1,1), n * n/nranks, MPI_FLOAT, &
    !                      label(:,:,1,1), n * n/nranks, MPI_FLOAT, &
    !                      MPI_COMM_WORLD, istat)
    ! !$acc end host_data

    !$acc wait ! Espera pelas comunicações MPI (REMOVIDO)

    ! Realizar inferência com batch size 1 (usando slicing 1:1)
    ! input e output são arrays globais, usamos a primeira posição do batch
    !$acc host_data use_device(input, output) if(simulation_device >= 0)
    istat = torchfort_inference("mymodel", input(:,:,1:1,1:1), output(:,:,1:1,1:1)) ! Presume que inference lida com batch size 1 via slicing
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    !$acc end host_data
    !$acc wait ! Espera a inferência terminar

    ! Calcular MSE - ajustado para o grid global (n, n)
    !$acc kernels if(simulation_device >= 0)
    mse = sum((label(:,:,1,1) - output(:,:,1,1))**2) / (real64(n)*n) ! <<< Ajustado divisor
    !$acc end kernels

    if (rank == 0 .and. mod(i-1, val_write_freq) == 0) then
      print*, "writing validation sample:", i, "mse:", mse
      write(idx,'(i7.7)') i
      filename = 'input_'//idx//'.h5'
      call write_sample(input(:,:,1,1), filename) ! Presume que write_sample lida com n,n
      filename = 'label_'//idx//'.h5'
      call write_sample(label(:,:,1,1), filename) ! Presume que write_sample lida com n,n
      filename = 'output_'//idx//'.h5'
      call write_sample(output(:,:,1,1), filename) ! Presume que write_sample lida com n,n
    endif
  end do
  !$acc end data ! Fim do data region para inferência

  ! salvar modelo e checkpoint (apenas rank 0, que agora é o único)
  if (rank == 0) then
    print*, "saving model and writing checkpoint..."
    istat = torchfort_save_model("mymodel", output_model_name)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    istat = torchfort_save_checkpoint("mymodel", output_checkpoint_dir)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  endif

  ! call MPI_Finalize(istat) ! REMOVIDO

end program train_sequential 
