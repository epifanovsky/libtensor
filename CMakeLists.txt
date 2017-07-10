cmake_minimum_required(VERSION 2.8.10)
project(LIBTENSOR)

set(LIBTENSOR_DIR ${LIBTENSOR_SOURCE_DIR})
get_filename_component(LIBTEST_DIR ${LIBTENSOR_DIR}/../libtest ABSOLUTE)
get_filename_component(LIBUTIL_DIR ${LIBTENSOR_DIR}/../libutil ABSOLUTE)
get_filename_component(LIBVMM_DIR ${LIBTENSOR_DIR}/../libvmm ABSOLUTE)

include(${LIBTENSOR_DIR}/cmake/CheckFortranSourceCompiles.cmake)
include(${LIBTENSOR_DIR}/cmake/CheckFortranSourceRuns.cmake)

if(IS_DIRECTORY ${LIBVMM_DIR})
    set(HAVE_LIBVMM TRUE)
else(IS_DIRECTORY ${LIBVMM_DIR})
    set(HAVE_LIBVMM FALSE)
    message(STATUS "Warning: libvmm was not found")
endif(IS_DIRECTORY ${LIBVMM_DIR})

if(LIBTENSOR_PACKAGE)
    include(${LIBUTIL_DIR}/cmake/DepsLibUtil.txt)
    include_directories("${LIBUTIL_DIR}")
    if(HAVE_LIBVMM)
        include(${LIBVMM_DIR}/cmake/DepsLibVmm.txt)
        include_directories("${LIBVMM_DIR}")
    endif(HAVE_LIBVMM)
else(LIBTENSOR_PACKAGE)
    include(${LIBUTIL_DIR}/cmake/SetupLibUtil.txt)
    if(HAVE_LIBVMM)
        include(${LIBVMM_DIR}/cmake/SetupLibVmm.txt)
    endif(HAVE_LIBVMM)
endif(LIBTENSOR_PACKAGE)
include(${LIBTENSOR_DIR}/cmake/DepsLibTensor.txt)

include_directories(${LIBTENSOR_DIR} ${ARMADILLO_INCLUDE_DIRS})

add_subdirectory(libtensor)
add_subdirectory(tests)
#add_subdirectory(performance_tests)

