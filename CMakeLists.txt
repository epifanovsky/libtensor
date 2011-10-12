cmake_minimum_required(VERSION 2.6)
project(LIBTENSOR)

set(LIBTENSOR_DIR ${LIBTENSOR_SOURCE_DIR})
get_filename_component(LIBTEST_DIR ${LIBTENSOR_DIR}/../libtest ABSOLUTE)
get_filename_component(LIBVMM_DIR ${LIBTENSOR_DIR}/../libvmm ABSOLUTE)

#Searching CUDA
#FIND_PACKAGE(CUDA)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ccbin icc -arch sm_13)

include(${LIBVMM_DIR}/cmake/SetupLibVmm.txt)
include(${LIBTENSOR_DIR}/cmake/DepsLibTensor.txt)

include_directories(${LIBTENSOR_DIR})

add_subdirectory(libtensor)
add_subdirectory(tests)
add_subdirectory(performance_tests)

