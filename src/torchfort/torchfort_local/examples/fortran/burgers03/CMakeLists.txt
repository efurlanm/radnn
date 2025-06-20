

add_executable(burgers03)
target_sources(burgers03
  PRIVATE
  burgers03.f90
)

target_include_directories(burgers03
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_Fortran_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(burgers03 PRIVATE MPI::MPI_Fortran ${CMAKE_BINARY_DIR}/lib/libtorchfort_fort.so ${CMAKE_BINARY_DIR}/lib/libtorchfort.so)

if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  target_compile_options(burgers03 PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${CUF_GPU_ARG}>)
  target_link_options(burgers03 PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: -acc -gpu=${CUF_GPU_ARG}>)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(burgers03 PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -fbackslash>)
endif()

install(
  TARGETS burgers03
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/burgers03
)