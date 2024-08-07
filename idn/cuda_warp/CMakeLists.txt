cmake_minimum_required(VERSION 3.10)

project(cuda_warp)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS})

include_directories("/usr/local/include/")
link_directories("/usr/local/lib/")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

CUDA_ADD_LIBRARY(cuda_utils_lib cuda_utils.cu OPTIONS -Xcompiler -fPIC)
# set_target_properties(cuda_utils_lib PROPERTIES POSITION_INDEPENDENT_CODE OFF)
pybind11_add_module(warp_event warp_event.cpp)
target_link_libraries(warp_event PRIVATE cuda_utils_lib)
