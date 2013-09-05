cmake_minimum_required(VERSION 2.6)
project(LIBTENSOR)

set(LIBTENSOR_DIR ${LIBTENSOR_SOURCE_DIR})
get_filename_component(LIBTEST_DIR ${LIBTENSOR_DIR}/../libtest ABSOLUTE)
get_filename_component(LIBUTIL_DIR ${LIBTENSOR_DIR}/../libutil ABSOLUTE)
get_filename_component(LIBVMM_DIR ${LIBTENSOR_DIR}/../libvmm ABSOLUTE)

if(IS_DIRECTORY ${LIBVMM_DIR})
    set(HAVE_LIBVMM TRUE)
else(IS_DIRECTORY ${LIBVMM_DIR})
    set(HAVE_LIBVMM FALSE)
    message(STATUS "Warning: libvmm was not found")
endif(IS_DIRECTORY ${LIBVMM_DIR})

include(${LIBUTIL_DIR}/cmake/SetupLibUtil.txt)
if(HAVE_LIBVMM)
    include(${LIBVMM_DIR}/cmake/SetupLibVmm.txt)
endif(HAVE_LIBVMM)
include(${LIBTENSOR_DIR}/cmake/DepsLibTensor.txt)

include_directories(${LIBTENSOR_DIR} ${CTF_INCLUDE_DIR})

add_subdirectory(ctf)
add_subdirectory(libtensor)
add_subdirectory(tests)
add_subdirectory(performance_tests)

