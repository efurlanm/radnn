find_package(HDF5 COMPONENTS Fortran REQUIRED)

set(fortran_example_targets
  burgers_train
  burgers_inference
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
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/burgers
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/burgers_model.pt
        ${CMAKE_CURRENT_SOURCE_DIR}/burgers_loss.pt
        ${CMAKE_CURRENT_SOURCE_DIR}/burgers_train_config.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_burgers_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/burgers_inference.py
        ${CMAKE_CURRENT_SOURCE_DIR}/compare_results.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/burgers)
