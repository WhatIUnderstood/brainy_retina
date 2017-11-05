cmake_minimum_required (VERSION 2.8.11)
project (CudaRetina)

set(CudaRetina_MAJOR_VERSION 0)
set(CudaRetina_MINOR_VERSION 1)
set(CudaRetina_PATCH_VERSION 0)
set(CudaRetina_VERSION
  ${CudaRetina_MAJOR_VERSION}.${CudaRetina_MINOR_VERSION}.${CudaRetina_PATCH_VERSION})

# Setup default installation directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX /opt/gpuretina CACHE PATH "default install path" FORCE )
endif()

## Find cuda on opencv dependencies
set(CUDA_SDK_ROOT_DIR /opt/cuda/cuda-8.0)
find_package(CUDA REQUIRED 8.0) # TODO minimal version for unified memory
find_package(OpenCV 3 COMPONENTS core imgproc highgui videoio HINTS /opt/opencv/3.2.0/ )#PATHS ${OPENCV_CMAKE_PATH})

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

## Set include path
set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/gpuretina)
endif()
set(INSTALL_CMAKE_DIR ${CMAKE_INSTALL_PREFIX}/${DEF_INSTALL_CMAKE_DIR})

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

## Get all sources
file(GLOB_RECURSE SRCS src/*.cpp)
file(GLOB_RECURSE CUDA_SRCS src/*.cu)

## Pass options to NVCC (cuda compiler)
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler  -lineinfo -arch=sm_60 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode  arch=compute_61,code=sm_61) #-gencode  arch=compute_61,code=sm_61 -rdynamic
#set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler  -lineinfo -gencode  arch=compute_61,code=sm_61) #-gencode  arch=compute_61,code=sm_61 -rdynamic
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

MACRO(INSTALL_HEADERS_WITH_DIRECTORY HEADER_LIST ROOT_PREFIX INCLUDE_INSTALL_PATH)
if( NOT "${ROOT_PREFIX}" STREQUAL "")
    string(REPLACE "/" "\\/" ROOT_EXTRA ${ROOT_PREFIX})
    message( FATAL "plop${ROOT_EXTRA}")
endif()
    FOREACH(HEADER ${${HEADER_LIST}})
        #STRING(REGEX MATCH "${ROOT_EXTRA}(.*)\\/" DIR ${HEADER}) #STRING(REGEX MATCH "(.\\*)\\[/\\]" DIR ${HEADER})
        STRING(REGEX MATCH "(.*)\\/" FULL_DIR ${HEADER}) #STRING(REGEX MATCH "(.\\*)\\[/\\]" DIR ${HEADER})
        STRING(REPLACE "${ROOT_PREFIX}" "" DIR ${FULL_DIR} )
        INSTALL(FILES ${HEADER} DESTINATION ${INCLUDE_INSTALL_PATH}/${DIR})
    ENDFOREACH(HEADER)

ENDMACRO(INSTALL_HEADERS_WITH_DIRECTORY)

set(HS "src/retinacuda.h;src/Cuda/retinastructs.h;src/Cuda/cuda_arrays.h;src/Cuda/declaration_helper.cuh")

INSTALL_HEADERS_WITH_DIRECTORY(HS "src/" "include/gpuretina")

#set_target_properties(gpuretina PROPERTIES
#  PUBLIC_HEADER "src/retinacuda.h;src/Cuda/retinastructs.h")

install(TARGETS gpuretina
    EXPORT gpuretinaTargets
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
)
#install(FILES ${CMAKE_BINARY_DIR}/libgpuretina.so DESTINATION "lib")
#install(DIRECTORY src/ DESTINATION "include/gpuretina" FILES_MATCHING PATTERN "*.h*")

### Uninstall the library ###

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/include/gpuretina
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_INSTALL_PREFIX}/lib/libgpuretina.so
)

### Build tests ###
set(OPENCV_CMAKE_PATH /opt/opencv/3.2.0/ )

set(OPENCV_ROOT_DIR /opt/opencv/3.2.0)

include_directories(gpuretina PUBLIC src/)

cuda_add_executable(gpuretina_test ${CUDA_SRCS} ${SRCS} tests/opencv/main.cpp)
TARGET_LINK_LIBRARIES(gpuretina_test ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})


### Force QT Creator to display all files
FILE(GLOB_RECURSE ALL_HEADERS_H "*.h")
FILE(GLOB_RECURSE ALL_HEADERS_CUH "src/Cuda/*.cuh")
add_custom_target(headers SOURCES ${ALL_HEADERS_H} ${ALL_HEADERS_CUH})

#Qt Creator make parameter to get errors parsed properly (nvcc output differ from gcc)
# make VERBOSE=1 3>&2 2>/tmp/error.out ; cat  /tmp/error.out | sed -r "s/\(([0-9]+)\)/:\1/g"  1>&2


##### Export ####
# Add all targets to the build-tree export set
export(TARGETS gpuretina
  FILE "${PROJECT_BINARY_DIR}/gpuretinaTargets.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE gpuretina)

#change variable to be absolute


# Create the FooBarConfig.cmake and FooBarConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}")
configure_file(gpuretinaConfig.cmake.in
  "${PROJECT_BINARY_DIR}/gpuretinaConfig.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${gpuretina_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(gpuretinaConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gpuretinaConfig.cmake" @ONLY)
# ... for both
configure_file(gpuretinaConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/gpuretinaConfigVersion.cmake" @ONLY)

# Install the FooBarConfig.cmake and FooBarConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gpuretinaConfig.cmake"
  "${PROJECT_BINARY_DIR}/gpuretinaConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree
install(EXPORT gpuretinaTargets DESTINATION
  "${INSTALL_CMAKE_DIR}")
