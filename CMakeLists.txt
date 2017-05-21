cmake_minimum_required (VERSION 2.8.11)
project (CudaRetina)

### Build the library ###

## Setup c++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Setup shared library flags
set (CMAKE_POSITION_INDEPENDENT_CODE TRUE)#FPIC

## Find opencv
#find_package(OpenCV 3)
#message(STATUS "${OpenCV_LIBRARIES}")
## Find cuda
set(CUDA_SDK_ROOT_DIR /opt/cuda/cuda-8.0)
#set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} /opt/Qt/Qt5.7.1/5.7/gcc_64/")
message(STATUS ${CMAKE_PREFIX_PATH})
find_package(CUDA REQUIRED 8.0) # TODO minimal version for unified memory

find_package(OpenCV 3 COMPONENTS core imgproc highgui videoio  )#PATHS ${OPENCV_CMAKE_PATH})

## Get all sources
file(GLOB_RECURSE SRCS src/*.cpp)
file(GLOB_RECURSE CUDA_SRCS src/*.cu)

## Pass options to NVCC (cuda compiler)
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler  -lineinfo -gencode arch=compute_61,code=sm_61) #-rdynamic
#set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_20,code=sm_21)

## Specify include directories
include_directories(
    kernels
    utility
    cuda
    src
    ${OpenCV_INCLUDE_DIRS}
)

## Build the library
cuda_add_library(gpuretina ${CUDA_SRCS} ${SRCS} SHARED) #${CUDA_SRCS}
TARGET_LINK_LIBRARIES(gpuretina ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

### Install the library ###

## Setup default installation directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX /opt/gpuretina CACHE PATH "default install path" FORCE )
endif()

install(FILES ${CMAKE_BINARY_DIR}/libgpuretina.so DESTINATION "lib")
install(DIRECTORY src/ DESTINATION "include/gpuretina" FILES_MATCHING PATTERN "*.h*")

### Uninstall the library ###

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/include/gpuretina
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_INSTALL_PREFIX}/lib/libgpuretina.so
)

### Build tests ###
set(OPENCV_CMAKE_PATH /opt/opencv/3.2.0/ )

set(OPENCV_ROOT_DIR /opt/opencv/3.2.0)

target_include_directories(gpuretina PUBLIC src/)

cuda_add_executable(gpuretina_test ${CUDA_SRCS} ${SRCS} tests/opencv/main.cpp)
TARGET_LINK_LIBRARIES(gpuretina_test ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})


### Force QT Creator to display all files
FILE(GLOB_RECURSE ALL_HEADERS_H "*.h")
FILE(GLOB_RECURSE ALL_HEADERS_CUH "src/Cuda/*.cuh")
add_custom_target(headers SOURCES ${ALL_HEADERS_H} ${ALL_HEADERS_CUH})

#Qt Creator make parameter to get errors parsed properly (nvcc output differ from gcc)
# make VERBOSE=1 3>&2 2>/tmp/error.out ; cat  /tmp/error.out | sed -r "s/\(([0-9]+)\)/:\1/g"  1>&2
