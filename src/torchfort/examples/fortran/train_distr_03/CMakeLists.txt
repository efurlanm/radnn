find_package(HDF5 COMPONENTS Fortran REQUIRED)


set(fortran_example_targets
#======================================
  train_distr_03
#======================================
)


#======================================
add_executable(train_distr_03)
target_sources(train_distr_03
  PRIVATE
  train_distr_03.f90
  simulation.f90
)
set_target_properties(train_distr_03 PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/2 )
#======================================


foreach(tgt ${fortran_example_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_Fortran_INCLUDE_DIRS}
    ${HDF5_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
  target_link_libraries(${tgt} PRIVATE hdf5::hdf5_fortran)
  target_link_libraries(${tgt} PRIVATE "${PROJECT_NAME}_fort")
  target_link_libraries(${tgt} PRIVATE ${PROJECT_NAME})
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${CUF_GPU_ARG}>)
    target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: -acc -gpu=${CUF_GPU_ARG}>)
  elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -fbackslash>)
  endif()
endforeach()


install(
  TARGETS ${fortran_example_targets}
#======================================
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/train_distr_03
#======================================
)


install(
#======================================
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config_mlp_native.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/config_fcn_torchscript.yaml
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/train_distr_03)
#======================================
