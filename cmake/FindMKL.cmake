#
#	Locates the Intel Math Kernel Library
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
	$ENV{HOME}/intel/Compiler/*/mkl/include
	/opt/intel/Compiler/*/mkl/include
	$ENV{HOME}/intel/mkl/*/include
	/opt/intel/mkl/*/include
)
if(MKL_INCLUDE_PATH)

set(MKL_FOUND TRUE)
get_filename_component(MKL_PATH ${MKL_INCLUDE_PATH}/.. ABSOLUTE)

#	MKL version is detected by the library binaries found
#
#	MKL pre-10 x86:    mkl_ia32
#	MKL pre-10 x86_64: mkl_em64t
#	MKL 10+ x86:       mkl_ia32 + mkl_core + mkl_intel
#	MKL 10+ x86_64:    mkl_ia32 + mkl_core + mkl_intel_lp64
#
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
	set(MKL_LIBRARY_PATH ${MKL_PATH}/lib/em64t)
	set(MKL_ARCH_A mkl_em64t)
	set(MKL_INTEL_A mkl_intel_lp64)
else(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
	set(MKL_LIBRARY_PATH ${MKL_PATH}/lib/32)
	set(MKL_ARCH_A mkl_ia32)
	set(MKL_INTEL_A mkl_intel)
endif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

find_library(MKL_CORE_A_PATH mkl_core PATHS ${MKL_LIBRARY_PATH})
find_library(MKL_ARCH_A_PATH ${MKL_ARCH_A} PATHS ${MKL_LIBRARY_PATH})
find_library(MKL_INTEL_A_PATH ${MKL_INTEL_A} PATHS ${MKL_LIBRARY_PATH})

if(MKL_ARCH_A_PATH)
	if(MKL_INTEL_A_PATH)
#		Version 10+
		set(MKL_LIBRARIES
			${MKL_INTEL_A_PATH} mkl_intel_thread mkl_core guide) 
	else(MKL_INTEL_A_PATH)
#		Version pre-10
		set(MKL_LIBRARIES
			${MKL_ARCH_A_PATH} guide)
	endif(MKL_INTEL_A_PATH)
endif(MKL_ARCH_A_PATH)

if(NOT MKL_FIND_QUIETLY)
	message(STATUS "Found Intel MKL: " ${MKL_PATH})
endif(NOT MKL_FIND_QUIETLY)

else(MKL_INCLUDE_PATH)

if(MKL_FIND_REQUIRED)
	message(FATAL_ERROR "Could not find Intel MKL")
endif(MKL_FIND_REQUIRED)

endif(MKL_INCLUDE_PATH)

