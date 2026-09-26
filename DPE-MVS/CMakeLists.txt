SET(CMAKE_C_COMPILER /usr/bin/gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required (VERSION 2.8)
project (DPE)

# Single-configuration generators otherwise compile the CPU stages without
# optimization when the user does not pass a build type.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

# recommand >=3.3.0
find_package(OpenCV REQUIRED)
# recommand >=1.62.0
find_package(Boost REQUIRED COMPONENTS filesystem system)
# recommand >=10.2.
find_package(CUDA REQUIRED )

include_directories(
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS})
include_directories(.)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++14 --compiler-options -Wall -gencode arch=compute_86,code=sm_86)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++14)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    # Keep CMake's optimized Release flags without host-side fast-math: the
    # adaptive geometry checks rely on finite-value and threshold comparisons.
endif()

cuda_add_executable(
    DPE
    main.h
    DPE.h
    DPE.cpp
    DPE.cu
    main.cpp
)

target_link_libraries(DPE
"${Boost_LIBRARIES}"
"${OpenCV_LIBS}"
)
