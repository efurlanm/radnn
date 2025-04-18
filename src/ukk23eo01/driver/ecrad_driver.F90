! ecrad_driver.F90 - Driver for offline ECRAD radiation scheme
!
! (C) Copyright 2014- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
!
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!
! Author:  Robin Hogan
! Email:   r.j.hogan@ecmwf.int
!
! ECRAD is the radiation scheme used in the ECMWF Integrated
! Forecasting System in cycle 43R3 and later. Several solvers are
! available, including McICA, Tripleclouds and SPARTACUS (the Speedy
! Algorithm for Radiative Transfer through Cloud Sides, a modification
! of the two-stream formulation of shortwave and longwave radiative
! transfer to account for 3D radiative effects). Gas optical
! properties are provided by the RRTM-G gas optics scheme.

! This program takes three arguments:
! 1) Namelist file to configure the radiation calculation
! 2) Name of a NetCDF file containing one or more atmospheric profiles
! 3) Name of output NetCDF file

! Modifications
!   2022-01-18  P. Ukkonen: Optional blocking of derived types, RRTMGP integration

program ecrad_driver

  ! --------------------------------------------------------
  ! Section 1: Declarations
  ! --------------------------------------------------------
  use parkind1,                 only : jprb, jprd ! Working/double precision

  use radiation_io,             only : nulout
  use radiation_interface,      only : setup_radiation, radiation, set_gas_units
  use radiation_config,         only : config_type, IGasModelRRTMGP, IGasModelRRTMGP_NN
  use radiation_single_level,   only : single_level_type
  use radiation_thermodynamics, only : thermodynamics_type
  use radiation_gas,            only : gas_type, &
       &   IVolumeMixingRatio, IMassMixingRatio, &
       &   IH2O, ICO2, IO3, IN2O, ICO, ICH4, IO2, ICFC11, ICFC12, &
       &   IHCFC22, ICCl4, GasName, GasLowerCaseName, NMaxGases
  use radiation_cloud,          only : cloud_type
  use radiation_aerosol,        only : aerosol_type
  use radiation_flux,           only : flux_type
  use radiation_save,           only : save_fluxes, save_net_fluxes, &
       &                               save_inputs, save_sw_diagnostics
  use radiation_general_cloud_optics, only : save_general_cloud_optics
  use ecrad_driver_config,      only : driver_config_type
  use ecrad_driver_read_input,  only : read_input 
  use ecrad_driver_read_input_blocked,  only : read_input_blocked, unblock_fluxes
  use easy_netcdf
#ifdef USE_TIMING
#ifndef USE_PAPI
  ! Timing library
 use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                  gptlpercent, gptloverhead, gptlpr_file
#endif
#endif
  implicit none

  ! Uncomment this if you want to use the "satur" routine below
!#include "satur.intfb.h"

#ifdef USE_PAPI  
#include "f90papi.h"
#include "gptl.inc"
#endif  

  ! The NetCDF file containing the input profiles
  type(netcdf_file)         :: file

  ! Derived types for the inputs to the radiation scheme
  type(config_type)         :: config
  type(single_level_type)   :: single_level
  type(thermodynamics_type) :: thermodynamics
  type(gas_type)            :: gas
  type(cloud_type)          :: cloud
  type(aerosol_type)        :: aerosol

  ! Configuration specific to this driver
  type(driver_config_type)  :: driver_config

  ! Derived type containing outputs from the radiation scheme
  type(flux_type)           :: flux

  ! Optional: blocking of derived types (more memory friends, resembles IFS use)
  type(single_level_type),    dimension(:), allocatable :: single_level_b
  type(thermodynamics_type),  dimension(:), allocatable :: thermodynamics_b
  type(gas_type),             dimension(:), allocatable :: gas_b
  type(cloud_type),           dimension(:), allocatable :: cloud_b
  type(aerosol_type),         dimension(:), allocatable :: aerosol_b
  type(flux_type),            dimension(:), allocatable :: flux_b

  integer :: ncol, nlev         ! Number of columns and levels
  integer :: istartcol, iendcol ! Range of columns to process

  ! Name of file names specified on command line
  character(len=512) :: file_name
  integer            :: istatus ! Result of command_argument_count

  ! For parallel processing of multiple blocks
  integer :: jblock, nblock, blocksize ! Block loop index and number

  ! Mapping matrix for shortwave spectral diagnostics
  real(jprb), allocatable :: sw_diag_mapping(:,:)
  
#ifndef NO_OPENMP
  ! OpenMP functions
  integer, external :: omp_get_thread_num
  real(kind=jprd), external :: omp_get_wtime
  ! Start/stop time in seconds
  real(kind=jprd) :: tstart, tstop
#endif

  ! For demonstration of get_sw_weights later on
  ! Ultraviolet weightings
  !integer    :: nweight_uv
  !integer    :: iband_uv(100)
  !real(jprb) :: weight_uv(100)
  ! Photosynthetically active radiation weightings
  !integer    :: nweight_par
  !integer    :: iband_par(100)
  !real(jprb) :: weight_par(100)

  ! Loop index for repeats (for benchmarking)
  integer :: jrepeat

  ! Are any variables out of bounds?
  logical :: is_out_of_bounds

!  integer    :: iband(20), nweights
!  real(jprb) :: weight(20)

#ifdef USE_TIMING
  integer :: ret
  integer values(8)
  character(len=100) :: timing_file_name, name_gas_model, name_solver
  character(5) :: prefix, suffix
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate

#ifdef USE_PAPI  
#ifdef PARKIND1_SINGLE
  ret = GPTLsetoption (PAPI_SP_OPS, 1);
#else
  ret = GPTLsetoption (PAPI_DP_OPS, 1);
#endif
! ret = GPTLsetoption (GPTL_IPC, 1);
ret = GPTLsetoption (PAPI_L1_DCM, 1);
ret = GPTLsetoption (GPTL_L3MRT, 1);
#endif  
  ret = gptlinitialize()
#endif

  ! --------------------------------------------------------
  ! Section 2: Configure
  ! --------------------------------------------------------

  ! Check program called with correct number of arguments
  if (command_argument_count() < 3) then
    stop 'Usage: ecrad config.nam input_file.nc output_file.nc'
  end if

  ! Use namelist to configure the radiation calculation
  call get_command_argument(1, file_name, status=istatus)
  if (istatus /= 0) then
    stop 'Failed to read name of namelist file as string of length < 512'
  end if

  ! Read "radiation" namelist into radiation configuration type
  call config%read(file_name=file_name)

  ! Read "radiation_driver" namelist into radiation driver config type
  call driver_config%read(file_name)

  ! Setup the radiation scheme: load the coefficients for gas and
  ! cloud optics, currently from RRTMG
  ! call setup_radiation(config)
  ! !!MOVED!! to after read_input because RRTMGP needs to know what gases are used 
  ! already when the coefficients are loaded

  ! Demonstration of how to get weights for UV and PAR fluxes
  !if (config%do_sw) then
  !  call config%get_sw_weights(0.2e-6_jprb, 0.4415e-6_jprb,&
  !       &  nweight_uv, iband_uv, weight_uv,&
  !       &  'ultraviolet')
  !  call config%get_sw_weights(0.4e-6_jprb, 0.7e-6_jprb,&
  !       &  nweight_par, iband_par, weight_par,&
  !       &  'photosynthetically active radiation, PAR')
  !end if

  ! --------------------------------------------------------
  ! Section 3: Read input data file
  ! --------------------------------------------------------

  ! Get NetCDF input file name
  call get_command_argument(2, file_name, status=istatus)
  if (istatus /= 0) then
    stop 'Failed to read name of input NetCDF file as string of length < 512'
  end if

  ! Open the file and configure the way it is read
  call file%open(trim(file_name), iverbose=driver_config%iverbose)

  ! Get NetCDF output file name
  call get_command_argument(3, file_name, status=istatus)
  if (istatus /= 0) then
    stop 'Failed to read name of output NetCDF file as string of length < 512'
  end if

  ! 2D arrays are assumed to be stored in the file with height varying
  ! more rapidly than column index. Specifying "true" here transposes
  ! all 2D arrays so that the column index varies fastest within the
  ! program.
  call file%transpose_matrices(.true.)

  if (driver_config%block_derived_types) then 
    call read_input_blocked(file, config, driver_config, nblock, ncol, nlev, &
         &          single_level_b, thermodynamics_b, &
         &          gas_b, cloud_b, aerosol_b)
         
    allocate(flux_b(nblock))
    ! All other derived type arrays are allocated inside read_input_blocked 
    blocksize = driver_config%nblocksize
  else
    call read_input(file, config, driver_config, ncol, nlev, &
          &         single_level, thermodynamics, &
          &         gas, cloud, aerosol)
  end if

  ! Close input file
  call file%close()

  ! Setup the radiation scheme: load the coefficients for gas and
  ! cloud optics, currently from RRTMG
  call setup_radiation(config)

  if (driver_config%iverbose >= 2) then
    write(nulout,'(a)') '-------------------------- OFFLINE ECRAD RADIATION SCHEME --------------------------'
    write(nulout,'(a)') 'Copyright (C) 2014- ECMWF'
    write(nulout,'(a)') 'Contact: Robin Hogan (r.j.hogan@ecmwf.int)'
#ifdef PARKIND1_SINGLE
    write(nulout,'(a)') 'Floating-point precision: single'
#else
    write(nulout,'(a)') 'Floating-point precision: double'
#endif
#ifdef USE_TIMING
    write(nulout,'(a)') 'Using General Purpose Timing Library'
#endif
    call config%print(driver_config%iverbose)
  end if

  ! Optionally compute shortwave spectral diagnostics in
  ! user-specified wavlength intervals
  if (driver_config%n_sw_diag > 0) then
    if (.not. config%do_surface_sw_spectral_flux) then
      stop 'Error: shortwave spectral diagnostics require do_surface_sw_spectral_flux=true'
    end if
    call config%get_sw_mapping(driver_config%sw_diag_wavelength_bound(1:driver_config%n_sw_diag+1), &
         &  sw_diag_mapping, 'user-specified diagnostic intervals')
    !if (driver_config%iverbose >= 3) then
    !  call print_matrix(sw_diag_mapping, 'Shortwave diagnostic mapping', nulout)
    !end if
  end if
  
  if (driver_config%do_save_aerosol_optics) then
    call config%aerosol_optics%save('aerosol_optics.nc', iverbose=driver_config%iverbose)
  end if

  if (driver_config%do_save_cloud_optics .and. config%use_general_cloud_optics) then
    call save_general_cloud_optics(config, 'hydrometeor_optics', iverbose=driver_config%iverbose)
  end if

  ! Compute seed from skin temperature residual
  ! if (driver_config%block_derived_types) then 
  !   do jblock = 1, nblock
  !     single_level_b(jblock)%iseed = int(1.0e9*(single_level_b(jblock)%skin_temperature &
  !           &                            -int(single_level_b(jblock)%skin_temperature)))
  !   end do
  ! else
  !   single_level%iseed = int(1.0e9*(single_level%skin_temperature &
  !         &                            -int(single_level%skin_temperature)))
  ! end if

  ! Set first and last columns to process
  if (driver_config%iendcol < 1 .or. driver_config%iendcol > ncol) then
    driver_config%iendcol = ncol
  end if

  if (driver_config%istartcol > driver_config%iendcol) then
    write(nulout,'(a,i0,a,i0,a,i0,a)') '*** Error: requested column range (', &
         &  driver_config%istartcol, &
         &  ' to ', driver_config%iendcol, ') is out of the range in the data (1 to ', &
         &  ncol, ')'
    stop 1
  end if
  
  ! Store inputs
  if (driver_config%do_save_inputs) then
    if (driver_config%block_derived_types) then 
       !write(nulout,'(a)') 'Warning: do_save_inputs ignored, inconsistent with block_derived_types'
    else
      call save_inputs('inputs.nc', config, single_level, thermodynamics, &
           &                gas, cloud, aerosol, &
           &                lat=spread(0.0_jprb,1,ncol), &
           &                lon=spread(0.0_jprb,1,ncol), &
           &                iverbose=driver_config%iverbose)
    end if
  end if

  ! --------------------------------------------------------
  ! Section 4: Call radiation scheme
  ! --------------------------------------------------------

  if (driver_config%block_derived_types) then 

    do jblock = 1, nblock
       if (driver_config%do_save_inputs) then
        !write(file_id, '(i0)') jblock
        !file_name_inp =  "inputs_" // trim(adjustl(file_id)) // ".nc"

        istartcol = (jblock-1) * driver_config%nblocksize &
          &    + driver_config%istartcol
        iendcol = min(istartcol + driver_config%nblocksize - 1, &
              &        driver_config%iendcol)

        write(file_name,'(a,i4.4,a,i4.4,a)') &
               &  'inputs_', istartcol, '-',iendcol,'.nc'
      
        call save_inputs(trim(file_name), config, single_level_b(jblock), thermodynamics_b(jblock), &
            &                gas_b(jblock), cloud_b(jblock), aerosol_b(jblock), &
            &                lat=spread(0.0_jprb,1,blocksize), &
            &                lon=spread(0.0_jprb,1,blocksize), &
            &                iverbose=driver_config%iverbose)
      end if
    
      ! Ensure the units of the gas mixing ratios are what is required
      ! by the gas absorption model
      call set_gas_units(config, gas_b(jblock))
  
      ! Compute saturation with respect to liquid (needed for aerosol
      ! hydration) call
      call thermodynamics_b(jblock)%calc_saturation_wrt_liquid(1, blocksize)
  
      ! Check inputs are within physical bounds, printing messagsae if not
      is_out_of_bounds =     gas_b(jblock)%out_of_physical_bounds(do_fix=driver_config%do_correct_unphysical_inputs) &
          & .or.   single_level_b(jblock)%out_of_physical_bounds(do_fix=driver_config%do_correct_unphysical_inputs) &
          & .or. thermodynamics_b(jblock)%out_of_physical_bounds(do_fix=driver_config%do_correct_unphysical_inputs) &
          & .or.          cloud_b(jblock)%out_of_physical_bounds(do_fix=driver_config%do_correct_unphysical_inputs) &
          & .or.        aerosol_b(jblock)%out_of_physical_bounds(do_fix=driver_config%do_correct_unphysical_inputs) 
  
      ! Allocate memory for the flux profiles, which may include arrays
      ! of dimension n_bands_sw/n_bands_lw, so must be called after
      ! setup_radiation
      call flux_b(jblock)%allocate(config, 1, blocksize, nlev)

    end do  

    call thermodynamics%allocate(ncol, nlev, allocated(thermodynamics_b(1)%h2o_sat_liq))

  else
  
    call set_gas_units(config, gas)

    ! Compute saturation with respect to liquid (needed for aerosol
    ! hydration) call
    call thermodynamics%calc_saturation_wrt_liquid(driver_config%istartcol,driver_config%iendcol)

    ! ...or alternatively use the "satur" function in the IFS (requires
    ! adding -lifs to the linker command line) but note that this
    ! computes saturation with respect to ice at colder temperatures,
    ! which is almost certainly incorrect
    !allocate(thermodynamics%h2o_sat_liq(ncol,nlev))
    !call satur(driver_config%istartcol, driver_config%iendcol, ncol, 1, nlev, .false., &
    !     0.5_jprb * (thermodynamics.pressure_hl(:,1:nlev)+thermodynamics.pressure_hl(:,2:nlev)), &
    !     0.5_jprb * (thermodynamics.temperature_hl(:,1:nlev)+thermodynamics.temperature_hl(:,2:nlev)), &
    !     thermodynamics%h2o_sat_liq, 2)

    ! Check inputs are within physical bounds, printing message if not
    is_out_of_bounds =     gas%out_of_physical_bounds(driver_config%istartcol, driver_config%iendcol, &
        &                                            driver_config%do_correct_unphysical_inputs) &
        & .or.   single_level%out_of_physical_bounds(driver_config%istartcol, driver_config%iendcol, &
        &                                            driver_config%do_correct_unphysical_inputs) &
        & .or. thermodynamics%out_of_physical_bounds(driver_config%istartcol, driver_config%iendcol, &
        &                                            driver_config%do_correct_unphysical_inputs) &
        & .or.          cloud%out_of_physical_bounds(driver_config%istartcol, driver_config%iendcol, &
        &                                            driver_config%do_correct_unphysical_inputs) &
        & .or.        aerosol%out_of_physical_bounds(driver_config%istartcol, driver_config%iendcol, &
        &                                            driver_config%do_correct_unphysical_inputs) 
  
  end if

  ! Allocate memory for the flux profiles, which may include arrays
  ! of dimension n_bands_sw/n_bands_lw, so must be called after
  ! setup_radiation
  call flux%allocate(config, 1, ncol, nlev)
  
  if (driver_config%iverbose >= 2) then
    write(nulout,'(a)')  'Performing radiative transfer calculations'
  end if
  
  ! Option of repeating calculation multiple time for more accurate
  ! profiling
#ifdef USE_TIMING
    ret =  gptlstart('radiation')
#endif   
#ifndef NO_OPENMP
  tstart = omp_get_wtime() 
#endif
  do jrepeat = 1,driver_config%nrepeat
    
    if (driver_config%do_parallel) then
      ! Run radiation scheme over blocks of columns in parallel
      
      ! Compute number of blocks to process
      if (driver_config%block_derived_types) then
        ! nblock already calculated, and in this case all columns in the inputed arrays are processed
      else 
          nblock = (driver_config%iendcol - driver_config%istartcol &
            &  + driver_config%nblocksize) / driver_config%nblocksize
      end if 

      !$OMP PARALLEL DO PRIVATE(istartcol, iendcol) SCHEDULE(DYNAMIC)
      do jblock = 1, nblock

        if (driver_config%block_derived_types) then

          call radiation(blocksize, nlev, 1, blocksize, config, &
                    &  single_level_b(jblock), thermodynamics_b(jblock), gas_b(jblock), cloud_b(jblock), &
                    aerosol_b(jblock), flux_b(jblock))

        else 
          ! Specify the range of columns to process.
          istartcol = (jblock-1) * driver_config%nblocksize &
          &    + driver_config%istartcol
          iendcol = min(istartcol + driver_config%nblocksize - 1, &
              &        driver_config%iendcol)

          if (driver_config%iverbose >= 3) then
#ifndef NO_OPENMP
          write(nulout,'(a,i0,a,i0,a,i0)')  'Thread ', omp_get_thread_num(), &
               &  ' processing columns ', istartcol, '-', iendcol
#else
          write(nulout,'(a,i0,a,i0)')  'Processing columns ', istartcol, '-', iendcol
#endif
          end if
          
          ! Call the ECRAD radiation scheme
          call radiation(ncol, nlev, istartcol, iendcol, config, &
              &  single_level, thermodynamics, gas, cloud, aerosol, flux)
          
        end if 

      end do
      !$OMP END PARALLEL DO
      
    else
      ! Run radiation scheme serially
      if (driver_config%iverbose >= 3) then
        write(nulout,'(a,i0,a)')  'Processing ', ncol, ' columns'
      end if
      
      ! Call the ECRAD radiation scheme
      call radiation(ncol, nlev, driver_config%istartcol, driver_config%iendcol, &
           &  config, single_level, thermodynamics, gas, cloud, aerosol, flux)
      
    end if
    
  end do
#ifdef USE_TIMING
  ret =  gptlstop('radiation')
#endif
#ifndef NO_OPENMP
  tstop = omp_get_wtime()
  write(nulout, '(a,g12.5,a)') 'Time elapsed in radiative transfer: ', tstop-tstart, ' seconds'
#endif


  if (driver_config%do_parallel .and. driver_config%block_derived_types) then
    !$OMP PARALLEL DO PRIVATE(istartcol, iendcol) SCHEDULE(RUNTIME)
    do jblock = 1, nblock
      call unblock_fluxes(jblock, blocksize, thermodynamics_b(jblock), flux_b(jblock), thermodynamics, flux)
    end do
  end if
  ! --------------------------------------------------------
  ! Section 5: Check and save output
  ! --------------------------------------------------------

  is_out_of_bounds = flux%out_of_physical_bounds(driver_config%istartcol, driver_config%iendcol)

  ! Store the fluxes in the output file
  if (.not. driver_config%do_save_net_fluxes) then
    call save_fluxes(file_name, config, thermodynamics, flux, &
         &   iverbose=driver_config%iverbose, is_hdf5_file=driver_config%do_write_hdf5, &
         &   experiment_name=driver_config%experiment_name, &
         &   is_double_precision=driver_config%do_write_double_precision)
  else
    call save_net_fluxes(file_name, config, thermodynamics, flux, &
         &   iverbose=driver_config%iverbose, is_hdf5_file=driver_config%do_write_hdf5, &
         &   experiment_name=driver_config%experiment_name, &
         &   is_double_precision=driver_config%do_write_double_precision)
  end if

  if (driver_config%n_sw_diag > 0) then
    ! Store spectral fluxes in user-defined intervals in a second
    ! output file
    call save_sw_diagnostics(driver_config%sw_diag_file_name, config, &
         &  driver_config%sw_diag_wavelength_bound(1:driver_config%n_sw_diag+1), &
         &  sw_diag_mapping, flux, iverbose=driver_config%iverbose, &
         &  is_hdf5_file=driver_config%do_write_hdf5, &
         &  experiment_name=driver_config%experiment_name, &
         &  is_double_precision=driver_config%do_write_double_precision)
  end if
  
  if (driver_config%iverbose >= 2) then
    write(nulout,'(a)') '------------------------------------------------------------------------------------'
  end if
  
#ifdef USE_TIMING
  ! End timers
  !  
  !! ret = gptlpr(driver_config%nblocksize)
  ! call date_and_time(values=values)
  ! write(timing_file_name,'(a,i4.4,a,i4.4,a,i4.4)') &
  ! & 'timing.',values(6),'_',values(7),'_',values(8)
  ! ret = gptlpr_file(trim(timing_file_name))
  call config%get_gas_optics_name(name_gas_model)
  call config%get_solver_name(name_solver)
  call date_and_time(values=values)
  write(timing_file_name,'(a,a,a,a,a,i0,a,i0,a,i4.4)') 'timing.', trim(name_gas_model), '_', trim(name_solver), &
    & '_block', driver_config%nblocksize, '_nrep', driver_config%nrepeat, '_', values(8)
  write(nulout,'(a,a)')  'Writing GPTL timing output to ', trim(timing_file_name)
  ret = gptlpr_file(trim(timing_file_name))
  ret = gptlfinalize()
#endif

end program ecrad_driver
