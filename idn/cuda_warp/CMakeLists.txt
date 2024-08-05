cmake_minimum_required(VERSION 3.10)

project(cuda_warp)

set(MY_PYBIND "/home/elendil/GD/idnet/idn/cuda_warp/thirdparty/pybind11")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


include_directories("/usr/local/include/")
link_directories("/usr/local/lib/")
add_subdirectory(${MY_PYBIND})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

CUDA_ADD_LIBRARY(cuda_utils_lib cuda_utils.cu)
pybind11_add_module(warp_event warp_event.cpp)
target_link_libraries(warp_event PRIVATE cuda_utils_lib)
