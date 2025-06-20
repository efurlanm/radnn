set(fortran_example_targets
  burgers
  burgers02
  burgers03
  burgers04
)

add_executable(burgers)
target_sources(burgers
  PRIVATE
  burgers.f90
)
set_target_properties(burgers PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(burgers02)
target_sources(burgers02
  PRIVATE
  burgers02.f90
)
set_target_properties(burgers02 PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(burgers03)
target_sources(burgers03
  PRIVATE
  burgers03.f90
)
set_target_properties(burgers03 PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(burgers04)
target_sources(burgers04
  PRIVATE
  burgers04.f90
)
set_target_properties(burgers04 PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

foreach(tgt ${fortran_example_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
  target_link_libraries(${tgt} PRIVATE "${PROJECT_NAME}_fort")
  target_link_libraries(${tgt} PRIVATE ${PROJECT_NAME})

  # Set RPATH for runtime library discovery
  target_link_options(${tgt} PRIVATE
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib
  )

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
