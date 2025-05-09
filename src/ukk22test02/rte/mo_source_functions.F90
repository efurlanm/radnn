! This code is part of Radiative Transfer for Energetics (RTE)
!
! Contacts: Robert Pincus and Eli Mlawer
! email:  rrtmgp@aer.com
!
! Copyright 2015-2018,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
! -------------------------------------------------------------------------------------------------
!
! Encapsulate source function arrays for longwave/lw/internal sources
!    and shortwave/sw/external source.
!
! -------------------------------------------------------------------------------------------------
module mo_source_functions
  use mo_rte_kind,      only: wp
  use mo_optical_props, only: ty_optical_props
  implicit none
  ! -------------------------------------------------------------------------------------------------
  !
  ! Type for longwave sources: computed at layer center, at layer edges using
  !   spectral mapping in each direction separately, and at the surface
  !
  type, extends(ty_optical_props), public :: ty_source_func_lw
    real(wp), allocatable, dimension(:,:,:) :: lev_source, lay_source
    real(wp), allocatable, dimension(:,:  ) :: sfc_source
    real(wp), allocatable, dimension(:,:  ) :: sfc_source_Jac ! surface source Jacobian 
    ! Optional output used for generating training data for neural networks
    real(wp), allocatable, dimension(:,:,:) :: planck_frac

  contains
    generic,   public :: alloc => alloc_lw, copy_and_alloc_lw
    procedure, private:: alloc_lw
    procedure, private:: copy_and_alloc_lw
    procedure, public :: is_allocated => is_allocated_lw
    procedure, public :: finalize => finalize_lw
    procedure, public :: get_subset => get_subset_range_lw
    procedure, public :: get_ncol => get_ncol_lw
    procedure, public :: get_nlay => get_nlay_lw
    ! validate?
  end type ty_source_func_lw
  ! -------------------------------------------------------------------------------------------------
  !
  ! Type for shortave sources: top-of-domain spectrally-resolved flux
  !
  type, extends(ty_optical_props), public :: ty_source_func_sw
    real(wp), allocatable, dimension(:,:  ) :: toa_source
  contains
    generic,   public :: alloc => alloc_sw, copy_and_alloc_sw
    procedure, private:: alloc_sw
    procedure, private:: copy_and_alloc_sw
    procedure, public :: is_allocated => is_allocated_sw
    procedure, public :: finalize => finalize_sw
    procedure, public :: get_subset => get_subset_range_sw
    procedure, public :: get_ncol => get_ncol_sw
    ! validate?
  end type ty_source_func_sw
  ! -------------------------------------------------------------------------------------------------
contains
  ! ------------------------------------------------------------------------------------------
  !
  !  Routines for initialization, validity checking, finalization
  !
  ! ------------------------------------------------------------------------------------------
  !
  ! Longwave
  !
  ! ------------------------------------------------------------------------------------------
  pure function is_allocated_lw(this)
    class(ty_source_func_lw), intent(in) :: this
    logical                              :: is_allocated_lw

    is_allocated_lw = this%is_initialized() .and. &
                      allocated(this%sfc_source)
  end function is_allocated_lw
  ! --------------------------------------------------------------
  function alloc_lw(this, ncol, nlay, save_pfrac) result(err_message)
    class(ty_source_func_lw),    intent(inout) :: this
    integer,                     intent(in   ) :: ncol, nlay
    logical,     optional,       intent(in   ) :: save_pfrac
    character(len = 128)                       :: err_message

    integer :: ngpt

    err_message = ""
    if(.not. this%is_initialized()) &
      err_message = "source_func_lw%alloc: not initialized so can't allocate"
    if(any([ncol, nlay] <= 0)) &
      err_message = "source_func_lw%alloc: must provide positive extents for ncol, nlay"
    if (err_message /= "") return

    !$acc enter data create(this)

    if(allocated(this%sfc_source)) deallocate(this%sfc_source)
    if(allocated(this%sfc_source_Jac)) deallocate(this%sfc_source_Jac)
    if(allocated(this%lev_source)) deallocate(this%lev_source)
    if(allocated(this%lay_source)) deallocate(this%lay_source)

    ngpt = this%get_ngpt()
    allocate(this%sfc_source(ngpt , ncol), this%lev_source(ngpt,nlay+1,ncol))
    allocate(this%sfc_source_Jac(ngpt, ncol), this%lay_source(ngpt,nlay,ncol))

    if(present(save_pfrac)) then
      if(save_pfrac) then
        if(allocated(this%planck_frac)) deallocate(this%planck_frac)
        allocate(this%planck_frac(ngpt,nlay,ncol))
        !$acc enter data create(this%planck_frac)
      end if
    end if
    
    !$acc enter data create(this%sfc_source, this%sfc_source_Jac, this%lev_source, this%lay_source)
  end function alloc_lw
  ! --------------------------------------------------------------
  function copy_and_alloc_lw(this, ncol, nlay, spectral_desc) result(err_message)
    class(ty_source_func_lw),    intent(inout) :: this
    integer,                     intent(in   ) :: ncol, nlay
    class(ty_optical_props ),    intent(in   ) :: spectral_desc
    character(len = 128)                       :: err_message

    err_message = ""
    if(.not. spectral_desc%is_initialized()) then
      err_message = "source_func_lw%alloc: spectral_desc not initialized"
      return
    end if
    call this%finalize()
    err_message = this%init(spectral_desc)
    if (err_message /= "") return
    err_message = this%alloc(ncol,nlay)
  end function copy_and_alloc_lw
  ! ------------------------------------------------------------------------------------------
  !
  ! Shortwave
  !
  ! ------------------------------------------------------------------------------------------
  pure function is_allocated_sw(this)
    class(ty_source_func_sw), intent(in) :: this
    logical                              :: is_allocated_sw

    is_allocated_sw = this%ty_optical_props%is_initialized() .and. &
                      allocated(this%toa_source)
  end function is_allocated_sw
  ! --------------------------------------------------------------
  function alloc_sw(this, ncol) result(err_message)
    class(ty_source_func_sw),    intent(inout) :: this
    integer,                     intent(in   ) :: ncol
    character(len = 128)                       :: err_message

    err_message = ""
    if(.not. this%is_initialized()) &
      err_message = "source_func_sw%alloc: not initialized so can't allocate"
    if(ncol <= 0) &
      err_message = "source_func_sw%alloc: must provide positive extents for ncol"
    if (err_message /= "") return

    if(allocated(this%toa_source)) deallocate(this%toa_source)

    allocate(this%toa_source(this%get_ngpt(),ncol))
    !$acc enter data create(this)
    !$acc enter data create(this%toa_source)
  end function alloc_sw
  ! --------------------------------------------------------------
  function copy_and_alloc_sw(this, ncol, spectral_desc) result(err_message)
    class(ty_source_func_sw),    intent(inout) :: this
    integer,                     intent(in   ) :: ncol
    class(ty_optical_props ),    intent(in   ) :: spectral_desc
    character(len = 128)                       :: err_message

    err_message = ""
    if(.not. spectral_desc%is_initialized()) then
      err_message = "source_func_sw%alloc: spectral_desc not initialized"
      return
    end if
    err_message = this%init(spectral_desc)
    if(err_message /= "") return
    err_message = this%alloc(ncol)
  end function copy_and_alloc_sw
  ! ------------------------------------------------------------------------------------------
  !
  ! Finalization (memory deallocation)
  !
  ! ------------------------------------------------------------------------------------------
  subroutine finalize_lw(this)
    class(ty_source_func_lw),    intent(inout) :: this

    if(allocated(this%lev_source)) then
      !$acc exit data delete(this%lev_source)
      deallocate(this%lev_source)
    end if
    if(allocated(this%lay_source)) then
      !$acc exit data delete(this%lay_source)
      deallocate(this%lay_source)
    end if

    if(allocated(this%sfc_source    )) then
      !$acc exit data delete(this%sfc_source)
      deallocate(this%sfc_source)
    end if 

    if(allocated(this%sfc_source_Jac    )) then
      !$acc exit data delete(this%sfc_source_Jac)
      deallocate(this%sfc_source_Jac)
    end if 

    if(allocated(this%planck_frac    )) then
      !$acc exit data delete(this%planck_frac)
      deallocate(this%planck_frac)
    end if 
    
    !$acc exit data delete(this)

  end subroutine finalize_lw
  ! --------------------------------------------------------------
  subroutine finalize_sw(this)
    class(ty_source_func_sw),    intent(inout) :: this

    if(allocated(this%toa_source    )) then
      !$acc exit data delete(this%toa_source)
      deallocate(this%toa_source)
    end if

    !$acc exit data delete(this)

  end subroutine finalize_sw
  ! ------------------------------------------------------------------------------------------
  !
  !  Routines for finding the problem size
  !
  ! ------------------------------------------------------------------------------------------
  pure function get_ncol_lw(this)
    class(ty_source_func_lw), intent(in) :: this
    integer :: get_ncol_lw

    if(this%is_allocated()) then
      get_ncol_lw = size(this%lev_source,3)
    else
      get_ncol_lw = 0
    end if
  end function get_ncol_lw
  ! --------------------------------------------------------------
  pure function get_nlay_lw(this)
    class(ty_source_func_lw), intent(in) :: this
    integer :: get_nlay_lw

    if(this%is_allocated()) then
      get_nlay_lw = size(this%lev_source,2) - 1
    else
      get_nlay_lw = 0
    end if
  end function get_nlay_lw
  ! --------------------------------------------------------------
  pure function get_ncol_sw(this)
    class(ty_source_func_sw), intent(in) :: this
    integer :: get_ncol_sw

    if(this%is_allocated()) then
      get_ncol_sw = size(this%toa_source,1)
    else
      get_ncol_sw = 0
    end if
  end function get_ncol_sw
  ! ------------------------------------------------------------------------------------------
  !
  !  Routines for subsetting
  !
  ! ------------------------------------------------------------------------------------------
  function get_subset_range_lw(full, start, n, subset) result(err_message)
    class(ty_source_func_lw), intent(inout) :: full
    integer,                  intent(in   ) :: start, n
    class(ty_source_func_lw), intent(inout) :: subset
    character(128)                          :: err_message

    err_message = ""
    if(.not. full%is_allocated()) then
      err_message = "source_func_lw%subset: Asking for a subset of unallocated data"
      return
    end if
    if(start < 1 .or. start + n-1 > full%get_ncol()) &
       err_message = "optical_props%subset: Asking for columns outside range"
    if(err_message /= "") return

    !
    ! Could check to see if subset is correctly sized, has consistent spectral discretization
    !
    if(subset%is_allocated()) call subset%finalize()
    err_message = subset%alloc(n, full%get_nlay(), full)
    if(err_message /= "") return
    subset%sfc_source(:,1:n)      = full%sfc_source    (:,start:start+n-1)
    subset%sfc_source_Jac(:,1:n)  = full%sfc_source_Jac(:,start:start+n-1)
    subset%lev_source(:,:,1:n)    = full%lev_source(:,:,start:start+n-1)
    subset%lay_source(:,:,1:n)    = full%lay_source(:,:,start:start+n-1)

  end function get_subset_range_lw
  ! ------------------------------------------------------------------------------------------
  function get_subset_range_sw(full, start, n, subset) result(err_message)
    class(ty_source_func_sw), intent(inout) :: full
    integer,                  intent(in   ) :: start, n
    class(ty_source_func_sw), intent(inout) :: subset
    character(128)                          :: err_message

    err_message = ""
    if(.not. full%is_allocated()) then
      err_message = "source_func_sw%subset: Asking for a subset of unallocated data"
      return
    end if
    if(start < 1 .or. start + n-1 > full%get_ncol()) &
       err_message = "optical_props%subset: Asking for columns outside range"
    if(err_message /= "") return

    !
    ! Could check to see if subset is correctly sized, has consistent spectral discretization
    !
    if(subset%is_allocated()) call subset%finalize()
    ! Seems like I should be able to call "alloc" generically but the compilers are complaining
    err_message = subset%copy_and_alloc_sw(n, full)

    subset%toa_source(:,1:n) = full%toa_source(:,start:start+n-1)
  end function get_subset_range_sw
end module mo_source_functions
