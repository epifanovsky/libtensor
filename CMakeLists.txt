cmake_minimum_required(VERSION 2.6)
project(LIBTENSOR)

set(LIBTEST_SOURCE_DIR ${LIBTENSOR_SOURCE_DIR}/../libtest/src)
set(LIBTEST_BINARY_DIR ${LIBTENSOR_SOURCE_DIR}/../libtest/build/src)
set(LIBVMM_SOURCE_DIR ${LIBTENSOR_SOURCE_DIR}/../libvmm/src)
set(LIBVMM_BINARY_DIR ${LIBTENSOR_SOURCE_DIR}/../libvmm/build/src)

set(SRC
	backtrace.C
	blas.C
	defs.C
	exception.C
	global_timings.C
	timer.C
	version.C
	btod/btod_delta_denom1.C
	btod/btod_delta_denom2.C
	btod/btod_mkdelta.C
	tod/tod_delta_denom1.C
	tod/tod_delta_denom2.C
	tod/tod_mkdelta.C
	tod/tod_solve.C
)

add_library(tensor ${SRC})

#	Detect whether the system is POSIX-compliant
if(UNIX)
	add_definitions(-DPOSIX)
endif(UNIX)

#       Detect the thread model
find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
	add_definitions(-DTHREAD=1)
	set(SYSTEM_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif(CMAKE_USE_PTHREADS_INIT)
if(CMAKE_USE_WIN32_THREADS_INIT)
	add_definitions(-DTHREAD=2)
	set(SYSTEM_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif(CMAKE_USE_WIN32_THREADS_INIT)


#
#	Detect the linear algebra library
#

set(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake)

#	Check for Intel MKL
find_package(MKL)
if(MKL_FOUND)
	set(HAVE_MKL TRUE)
endif(MKL_FOUND)

#	Check for CBLAS
find_package(CBLAS)
if(CBLAS_FOUND)
	set(HAVE_CBLAS TRUE)
endif(CBLAS_FOUND)

set(HAVE_CLAPACK FALSE)
set(HAVE_LAPACK FALSE)

#	Set up compilation flags and paths
if(HAVE_MKL)
	add_definitions(-DUSE_MKL)
	include_directories(${MKL_INCLUDE_PATH})
	set(BLAS_LIBS ${MKL_LIBRARIES})
else(HAVE_MKL)
if(HAVE_CBLAS)
	add_definitions(-DUSE_CBLAS)
	set(BLAS_LIBS ${CBLAS_LIBRARIES})
endif(HAVE_CBLAS)
endif(HAVE_MKL)


set(LIBS ${BLAS_LIBS} ${SYSTEM_LIBS})

include_directories(${LIBTEST_SOURCE_DIR})
add_library(test STATIC IMPORTED)
set_target_properties(test PROPERTIES
	IMPORTED_LOCATION ${LIBTEST_BINARY_DIR}/libtest.a
)

include_directories(${LIBVMM_SOURCE_DIR})
add_library(vmm STATIC IMPORTED)
set_target_properties(vmm PROPERTIES
	IMPORTED_LOCATION ${LIBVMM_BINARY_DIR}/libvmm.a
)

include_directories(${LIBTENSOR_SOURCE_DIR})

add_subdirectory(tests)
#add_subdirectory(performance_tests)

