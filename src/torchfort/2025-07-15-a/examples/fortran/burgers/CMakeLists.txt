find_package(HDF5 COMPONENTS Fortran REQUIRED)

set(fortran_example_targets
  burgers_train
  burgers_inference
  burgers_inference_test
  simple_inference
)

add_executable(burgers_train)
target_sources(burgers_train
  PRIVATE
  burgers_train.f90
)
set_target_properties(burgers_train PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(burgers_inference)
target_sources(burgers_inference
  PRIVATE
  burgers_inference.f90
)
set_target_properties(burgers_inference PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(burgers_inference_test)
target_sources(burgers_inference_test
  PRIVATE
  burgers_inference_test.f90
)
set_target_properties(burgers_inference_test PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(simple_inference)
target_sources(simple_inference
  PRIVATE
  simple_inference.f90
)
set_target_properties(simple_inference PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

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
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${CUF_GPU_ARG} -g -traceback>)
    target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: -acc -gpu=${CUF_GPU_ARG}>)
  elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -fbackslash>)
  endif()
endforeach()

install(
  TARGETS ${fortran_example_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/burgers
)
