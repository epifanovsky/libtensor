cmake_minimum_required(VERSION 3.0)
project(LIBTENSOR CXX)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(psi4OptionsTools)
include(SetupCompiler)

option_with_print(BUILD_SHARED_LIBS "Build shared libraries instead of static ones (CMake default is static)" ON)
option_with_print(ENABLE_OPENMP "Enables OpenMP parallelization" ON)
option_with_print(INSTALL_DEVEL_HEADERS "Install devel headers" ON)
option_with_default(CMAKE_BUILD_TYPE "Build type (Release or Debug)" Release)
option_with_default(BUILD_FPIC "Compile static libraries with position independent code" ON)
option_with_default(CMAKE_INSTALL_LIBDIR "Directory to which libraries installed" lib)

if(UNIX)
    add_definitions(-DPOSIX)
endif(UNIX)

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# pthreads
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads)
if (NOT CMAKE_USE_PTHREADS_INIT)
	message(FATAL_ERROR "We need pthreads as the threading backend at the moment")
endif()
add_definitions(-DUSE_PTHREADS)

# BLAS
# find_package(LAPACK REQUIRED)
# set(BLAS_LAPACK_INCLUDE "")
# set(BLAS_LAPACK_LINKER_FLAGS ${LAPACK_LINKER_FLAGS})
# set(BLAS_LAPACK_LIBRARIES ${LAPACK_LIBRARIES})
option(BLA_STATIC "Statically link to BLAS and LAPACK" OFF)
if (UNIX AND NOT APPLE)
	set(BLA_VENDOR "OpenBLAS" CACHE STRING "BLAS Vendor to use (see CMake documentation)")
elseif (APPLE)
	set(BLA_VENDOR "Apple" CACHE STRING "BLAS Vendor to use (see CMake documentation)")
endif()
find_package(BLAS REQUIRED)
include_directories(cblas)
add_definitions(-DUSE_CBLAS)

if    (BLA_VENDOR STREQUAL "Apple")
elseif(BLA_VENDOR STREQUAL "OpenBLAS")
elseif(BLA_VENDOR STREQUAL "Intel10_64lp_seq" OR BLA_VENDOR STREQUAL "Intel10_64lp")
	# Find the directory with mkl.h
	list(GET BLAS_LIBRARIES 1 BLAS_LIB)
	get_filename_component(BLAS_DIR ${BLAS_LIB} DIRECTORY)
	find_file(MKLINCLUDE mkl.h HINTS "${BLAS_DIR}/../include" "${BLAS_DIR}/../../include")
	get_filename_component(BLAS_INCLUDE_DIR ${MKLINCLUDE} DIRECTORY)
else()
	message(WARNING "BLAS vendor ${BLA_VENDOR} has not been tested with adccore")
endif()
set(BLAS_LAPACK_LIBRARIES ${BLAS_LIBRARIES})


# include(${LIBUTIL_SOURCE_DIR}/cmake/SetupLibUtil.txt)
set(LIBUTIL_SOURCE_DIR ${LIBTENSOR_SOURCE_DIR})
set(LIBTENSOR_DIR ${LIBTENSOR_SOURCE_DIR})
set(LIBUTIL_DIR ${LIBUTIL_SOURCE_DIR})

# include(cmake/DepsLibTensor.txt)

add_subdirectory(libutil)
add_subdirectory(libtensor)

enable_testing()
option(LIBTENSOR_TESTS "Build libtensor tests" ON)
if (LIBTENSOR_TESTS)
    add_subdirectory(tests)
    # add_subdirectory(performance_tests)
endif()
