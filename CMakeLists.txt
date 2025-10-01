cmake_minimum_required(VERSION 3.15)
project(Bratu2D_NK_MPIOMP CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build tests" ON)

find_package(MPI REQUIRED)

add_library(bratu
    src/options.cpp
    src/grid.cpp
    src/bratu.cpp
    src/gmres.cpp
    src/newton.cpp
    src/linalg.cpp
    src/io.cpp
)
target_include_directories(bratu PUBLIC src)
target_link_libraries(bratu PUBLIC MPI::MPI_CXX)

if (MSVC)
  # Option 1: MSVC front-end with SIMD (works broadly)
  target_compile_options(bratu PUBLIC /openmp:experimental)

  # Option 2 (alternative): LLVM OpenMP mode, if you’re actually using clang-cl.
  # For cl.exe this is often flaky; prefer :experimental with cl.exe.
  # target_compile_options(bratu PUBLIC /openmp:llvm)
else()
  find_package(OpenMP REQUIRED)
  target_link_libraries(bratu PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(bratu2d src/main.cpp)
target_link_libraries(bratu2d PRIVATE bratu)

if(BUILD_TESTS)
  add_executable(fd_check tests/fd_check.cpp)
  target_link_libraries(fd_check PRIVATE bratu)
endif()
