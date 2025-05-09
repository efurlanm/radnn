!train.f90

subroutine print_help_message
  print*, &
  "Usage: train [options]\n"// &
  "options:\n"// &
  "\t--configfile\n" // &
  "\t\tTorchFort configuration file to use. (default: config_mlp_native.yaml) \n" // &
  "\t--simulation_device\n" // &
  "\t\tDevice to run simulation on. (-1 for CPU, >= 0 for GPU by index. default: 0) \n" // &
  "\t--train_device\n" // &
  "\t\tDevice to run model training/inference on. (-1 for CPU, >= 0 for GPU by index. default: 0) \n" // &
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

program train
  use, intrinsic :: iso_fortran_env, only: real32, real64
#ifdef _OPENACC
  use openacc
#endif
  use simulation

!======================================
  use torchfort
!======================================

  implicit none

  integer :: i, j, istat
  integer :: n, nchannels, batch_size
  real(real32) :: a(2), dt
  real(real32) :: loss_val
  real(real64) :: mse
  real(real32), allocatable :: u(:,:), u_div(:,:)
  real(real32), allocatable :: input(:,:,:,:), label(:,:,:,:), output(:,:,:,:)
  character(len=7) :: idx
  character(len=256) :: filename
  logical :: load_ckpt = .false.
  integer :: train_step_ckpt = 0
  integer :: val_step_ckpt = 0
#ifdef _OPENACC
  integer(acc_device_kind) :: dev_type
#endif

  ! command line arguments
  character(len=256) :: configfile = "config_mlp_native.yaml"
  character(len=256) :: checkpoint_dir
  character(len=256) :: output_model_name = "model.pt"
  character(len=256) :: output_checkpoint_dir = "checkpoint"
  integer :: ntrain_steps = 100000
  integer :: nval_steps = 1000
  integer :: val_write_freq = 10
  integer :: model_device = 0
  integer :: simulation_device = 0

  logical :: skip_next
  character(len=256) :: arg

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
        if (model_device < -1) then
          print*, "Invalid train device type argument."
          call exit(1)
        endif
        skip_next = .true.
      case('--simulation_device')
        call get_command_argument(i+1, arg)
        read(arg, *) simulation_device
        if (simulation_device < -1) then
          print*, "Invalid simulation device type argument."
          call exit(1)
        endif
        skip_next = .true.
      case('-h')
        call print_help_message
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
  if (simulation_device >= 0) then
    dev_type = acc_get_device_type()
    call acc_set_device_num(simulation_device, dev_type)
    call acc_init(dev_type)
  endif
#endif

  print*, "Run settings:"
  print*, "\tconfigfile: ", trim(configfile)
  if (simulation_device == TORCHFORT_DEVICE_CPU) then
    print*, "\tsimulation_device: cpu"
  else
    print*, "\tsimulation_device: gpu ", simulation_device
  endif
  if (model_device == TORCHFORT_DEVICE_CPU) then
    print*, "\ttrain_device: cpu"
  else
    print*, "\ttrain_device: gpu ", model_device
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

! De: https://github.com/NVIDIA/TorchFort/tree/master/examples/fortran/simulation :

!A simulação usa uma grade numérica de $32^2$ pontos com precisão simples (o que explica o n = 32)

! "NCHW" é um formato comum utilizado para organizar os dados em redes neurais convolucionais (CNNs) e frameworks de aprendizado de máquina. Ele descreve a ordem dos eixos ou dimensões dos tensores, que são normalmente usados para representar imagens ou dados multidimensionais. Cada letra em "NCHW" refere-se a uma dimensão específica:
! N: Número de amostras ou imagens no lote (batch size). Esse eixo é usado para armazenar múltiplas entradas ao mesmo tempo.
! C: Número de canais. Por exemplo, para imagens coloridas, geralmente são 3 canais (vermelho, verde e azul - RGB), e para imagens em escala de cinza, é 1 canal.
! H: Altura da entrada, ou seja, a quantidade de pixels verticalmente.
! W: Largura da entrada, ou seja, a quantidade de pixels horizontalmente.

! **batch_size = 16** : o lote terá 16 amostras (ou seja, 16 conjuntos de dados de entrada, input, e seus respectivos rótulos ou saídas esperadas, label).

! **dt = 0.01** : For this example, we use the analytical solution as a stand-in for a real numerical solver, sequentially generating $u(x,y,t)$ samples at $\Delta t = 0.01$ intervals in time over the course of training. We compute corresponding analytical divergence fields, $\nabla \cdot (\mathbf{a} u(x,y,t))$, to use as training labels.

! **a = [1.0, 0.789]** : $a_x = 1.0$ e $a_y = 0.789$ são as componentes do vetor $\mathbf{a}$ em relação aos eixos x e y, respectivamente. Esse vetor descreve a direção e a magnitude da advecção (movimento de transporte) do campo escalar $u(x, y, t)$ na simulação.

  ! model/simulation parameters
  n = 32    !!! define o tamanho do numerical grid (input, label, output)
  nchannels = 1
  batch_size = 16
  dt = 0.01
  a = [1.0, 0.789] ! off-angle to generate more varied training data

  
! simulation.f90 defines a Fortran module to control the simulation. It has an initialization suboutine and a subroutine to advance the solution in time, returning both the scalar solution field, `u`, and corresponding divergence field, `u_div`. 
  
  ! allocate "simulation" data
  allocate(u(n, n))
  allocate(u_div(n, n))

  ! allocate training/inference data in standard 2D layout (NCHW, row-major)
  allocate(input(n, n, nchannels, batch_size))
  allocate(label(n, n, nchannels, batch_size))
  allocate(output(n, n, nchannels, batch_size))




!======================================
  ! set torch benchmark mode
  istat = torchfort_set_cudnn_benchmark(.true.)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! setup the model
  istat = torchfort_create_model("mymodel", configfile, model_device)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop

  ! load training checkpoint if requested
  if (load_ckpt) then
    print*, "loading checkpoint..."
    istat = torchfort_load_checkpoint("mymodel", checkpoint_dir, train_step_ckpt, val_step_ckpt)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  endif
!======================================



  call init_simulation(n, dt, a, train_step_ckpt*batch_size*dt, 0, 1, simulation_device)





! Durante o loop de treinamento, o modelo recebe o campo escalar u (armazenado em input) e o campo de divergência analítico (armazenado em label). O modelo aprende a mapear o input para o label, ajustando seus parâmetros para que sua previsão (output) se aproxime do label.
  
  !--------------------------------------
  ! run training
  !--------------------------------------
  if (ntrain_steps >= 1) print*, "start training..."
  !$acc data copyin(u, u_div, input, label) if(simulation_device >= 0)
  do i = 1, ntrain_steps


    do j = 1, batch_size
      call run_simulation_step(u, u_div)
      !$acc kernels async if(simulation_device >= 0)
      input(:,:,1,j) = u
      label(:,:,1,j) = u_div
      !$acc end kernels
    end do
    !$acc wait
    !$acc host_data use_device(input, label) if(simulation_device >= 0)
    !======================================
    istat = torchfort_train("mymodel", input, label, loss_val)
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
    !======================================
    !$acc end host_data
    !$acc wait


  end do
  !$acc end data
  !--------------------------------------

  if (ntrain_steps >= 1) print*, "final training loss: ", loss_val
  


! No loop de validação, o label continua a ser gerado a partir da simulação. Após o modelo realizar a inferência (previsão) com o input correspondente, o resultado do modelo (output) é comparado com o label para calcular métricas de erro, como o Mean Squared Error (MSE).

!$acc data ... ! $acc end data: Define uma região de dados OpenACC
! copyin(u, u_div, input, label): indica que os arrays u, u_div, input e label devem ser copiados da memória do hospedeiro para a memória do dispositivo quando a região de dados é iniciada.
! copyout(output): indica que o array output deve ser copiado da memória do dispositivo de volta para a memória do hospedeiro quando a região de dados é finalizada.

  !--------------------------------------
  ! run inference
  !--------------------------------------
  if (nval_steps >= 1) print*, "start validation..."
  !$acc data copyin(u, u_div, input, label) copyout(output) if(simulation_device >= 0)
  do i = 1, nval_steps
    call run_simulation_step(u, u_div)
    !$acc kernels if(simulation_device >= 0) async
    input(:,:,1,1) = u
    label(:,:,1,1) = u_div
    !$acc end kernels
    !$acc wait

    !$acc host_data use_device(input, output) if(simulation_device >= 0)
!======================================
    istat = torchfort_inference("mymodel", input(:,:,1:1,1:1), output(:,:,1:1,1:1))
    if (istat /= TORCHFORT_RESULT_SUCCESS) stop
!======================================
    !$acc end host_data
    !$acc wait

    !$acc kernels if(simulation_device >= 0)
    mse = sum((label(:,:,1,1) - output(:,:,1,1))**2) / (n*n)
    !$acc end kernels

    if (mod(i-1, val_write_freq) == 0) then
      print*, "writing validation sample:", i, "mse:", mse
      write(idx,'(i7.7)') i
      filename = 'data/input_'//idx//'.h5'
      call write_sample(input(:,:,1,1), filename)
      filename = 'data/label_'//idx//'.h5'
      call write_sample(label(:,:,1,1), filename)
      filename = 'data/output_'//idx//'.h5'
      call write_sample(output(:,:,1,1), filename)
    endif
  end do
  !$acc end data
  !--------------------------------------



!======================================
  print*, "saving model and writing checkpoint..."
  istat = torchfort_save_model("mymodel", output_model_name)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
  istat = torchfort_save_checkpoint("mymodel", output_checkpoint_dir)
  if (istat /= TORCHFORT_RESULT_SUCCESS) stop
!======================================


end program train
