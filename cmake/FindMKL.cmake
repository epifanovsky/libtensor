#
#	Detects the presence of the Intel Math Kernel Library
#
#	Output:
#
#	MKL_FOUND - TRUE/FALSE - Whether the library has been found.
#		If FALSE, all other output variables are not defined.
#
#	MKL_PATH         - Library home.
#	MKL_INCLUDE_PATH - Path to the library's header files.
#	MKL_LIBRARY_PATH - Path to the library's binaries.
#	MKL_LIBRARIES    - Line for the linker.
#	
set(MKL_FOUND FALSE)

find_path(MKL_INCLUDE_PATH
	NAMES
	mkl.h
	PATHS
	$ENV{MKLROOT}/include
	/opt/intel/mkl/*/include
)
if(MKL_INCLUDE_PATH)

set(MKL_FOUND TRUE)
get_filename_component(MKL_PATH ${MKL_INCLUDE_PATH}/.. ABSOLUTE)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
	find_library(MKL_LIBRARY_PATH
		mkl_intel_lp64
		PATHS ${MKL_PATH}/lib/em64t)
	if(MKL_LIBRARY_PATH)
		set(MKL_LIBRARIES
			${MKL_LIBRARY_PATH}
			mkl_intel_thread mkl_core guide) 
	endif(MKL_LIBRARY_PATH)
else(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
	find_library(MKL_LIBRARY_PATH
		mkl_intel
		PATHS ${MKL_PATH}/lib/32)
	if(MKL_LIBRARY_PATH)
		set(MKL_LIBRARIES
			${MKL_LIBRARY_PATH}
			mkl_intel_thread mkl_core guide) 
	endif(MKL_LIBRARY_PATH)
endif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

message(STATUS "Found Intel MKL: " ${MKL_PATH})

endif(MKL_INCLUDE_PATH)

