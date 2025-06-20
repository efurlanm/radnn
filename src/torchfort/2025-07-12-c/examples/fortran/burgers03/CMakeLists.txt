find_package(HDF5 COMPONENTS Fortran REQUIRED)

set(fortran_example_targets
  burgers03
  test_load
  test_torchfort
)

add_executable(test_torchfort)
  target_sources(test_torchfort
  PRIVATE
  test_torchfort.f90
)

add_executable(burgers03)
  target_sources(burgers03
  PRIVATE
  burgers03.f90
  simulation.f90
)
set_target_properties(burgers03 PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(test_load)
  target_sources(test_load
  PRIVATE
  test_load.f90
)





foreach(tgt ${fortran_example_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${TORCHFORT_INCLUDE_DIRS}
    ${MPI_Fortran_INCLUDE_DIRS}
    ${HDF5_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
  target_link_libraries(${tgt} PRIVATE hdf5::hdf5_fortran)
  target_link_libraries(${tgt} PRIVATE torchfort_fort)
  target_link_libraries(${tgt} PRIVATE torchfort)
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${CUF_GPU_ARG} -g>)
    target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: -acc -gpu=${CUF_GPU_ARG} -g>)
  elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -fbackslash>)
  endif()
endforeach()

install(
  TARGETS ${fortran_example_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/burgers03
)


install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config_mlp_native.yaml
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/burgers03)