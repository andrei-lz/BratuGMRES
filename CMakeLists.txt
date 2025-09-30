cmake_minimum_required(VERSION 3.15)
project(Bratu2D_NK_MPIOMP CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build tests" ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_library(bratu
    src/options.cpp
    src/grid.cpp
    src/bratu.cpp
    src/gmres.cpp
    src/newton.cpp
    src/linalg.cpp
    src/io.cpp
)
target_include_directories(bratu PUBLIC include)
target_link_libraries(bratu PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(bratu2d src/main.cpp)
target_link_libraries(bratu2d PRIVATE bratu)

if(BUILD_TESTS)
  add_executable(fd_check tests/fd_check.cpp)
  target_link_libraries(fd_check PRIVATE bratu)
endif()
