cmake_minimum_required(VERSION 2.6)
project(LIBTENSOR)

set(LIBTENSOR_DIR ${LIBTENSOR_SOURCE_DIR})
get_filename_component(LIBTEST_SOURCE_DIR
	${LIBTENSOR_SOURCE_DIR}/../libtest/src
	ABSOLUTE)
get_filename_component(LIBTEST_BINARY_DIR
	${LIBTENSOR_SOURCE_DIR}/../libtest/build/src
	ABSOLUTE)
get_filename_component(LIBVMM_SOURCE_DIR
	${LIBTENSOR_SOURCE_DIR}/../libvmm/src
	ABSOLUTE)
get_filename_component(LIBVMM_BINARY_DIR
	${LIBTENSOR_SOURCE_DIR}/../libvmm/build/src
	ABSOLUTE)

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

include(CMakeEnv.txt)

include_directories(${LIBTENSOR_SOURCE_DIR})

add_subdirectory(tests)
add_subdirectory(performance_tests)

