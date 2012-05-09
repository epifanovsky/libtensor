#
#    Locates the Intel Math Kernel Library
#
#    Output:
#
#    MKL_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    MKL_PATH         - Library home.
#    MKL_INCLUDE_PATH - Path to the library's header files.
#    MKL_LIBRARY_PATH - Path to the library's binaries.
#    MKL_LIBRARIES    - Line for the linker.
#
#    The following locations are searched:
#    1. CMake MKLROOT
#    2. Environment MKLROOT
#    3. LD_LIBRARY_PATH/DYLD_LIBRARY_PATH/PATH
#    4. Default MKL installation directories
#    
set(MKL_FOUND FALSE)

#
#    Set up search locations
#
if(APPLE)
    set(LD_LIBRARY_PATH_NAME DYLD_LIBRARY_PATH)
else(APPLE)
if(WIN32)
    set(LD_LIBRARY_PATH_NAME PATH)
else(WIN32)
    set(LD_LIBRARY_PATH_NAME LD_LIBRARY_PATH)
endif(WIN32)
endif(APPLE)

if(MKLROOT)
    find_path(MKL_PATH_MKLROOT mkl.h PATHS ${MKLROOT}/include)
endif(MKLROOT)
if(NOT MKL_PATH_MKLROOT)
    find_path(MKL_PATH_MKLROOT mkl.h PATHS $ENV{MKLROOT}/include)
endif(NOT MKL_PATH_MKLROOT)

find_library(MKL_PATH_LD NAMES mkl mkl_core PATHS ENV ${LD_LIBRARY_PATH_NAME})

find_path(MKL_PATH_GUESS mkl.h
    PATHS
    $ENV{HOME}/intel/Compiler/*/mkl/include
    /opt/intel/Compiler/*/mkl/include
    $ENV{HOME}/intel/mkl/*/include
    /opt/intel/mkl/*/include)

if(MKL_PATH_MKLROOT)
    get_filename_component(MKL_PATH ${MKL_PATH_MKLROOT}/.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(MKL_PATH_MKLROOT)
if(NOT MKL_FOUND AND MKL_PATH_LD)
    get_filename_component(MKL_PATH ${MKL_PATH_LD}/../../.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_LD)
if(NOT MKL_FOUND AND MKL_PATH_GUESS)
    get_filename_component(MKL_PATH ${MKL_PATH_GUESS}/.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_GUESS)

if(MKL_FOUND)

if(APPLE)
    set(MKL_INCLUDE_PATH ${MKL_PATH}/Headers)
    set(MKL_LIB_PATH ${MKL_PATH}/Versions/Current/lib)
else(APPLE)
    set(MKL_INCLUDE_PATH ${MKL_PATH}/include)
    set(MKL_LIB_PATH ${MKL_PATH}/lib)
endif(APPLE)

#
#    MKL version is detected by the library binaries found
#
#    MKL pre-10 x86:    mkl_ia32
#    MKL pre-10 x86_64: mkl_em64t
#    MKL 10+ x86:       mkl_ia32 + mkl_core + mkl_intel
#    MKL 10+ x86_64:    mkl_em64t + mkl_core + mkl_intel_lp64
#
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/em64t)
    set(MKL_ARCH_A mkl_em64t)
    set(MKL_INTEL_A mkl_intel_lp64)
    set(MKL_SOLVER_A mkl_solver_lp64)
else(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/32)
    set(MKL_ARCH_A mkl_ia32)
    set(MKL_INTEL_A mkl_intel)
    set(MKL_SOLVER_A mkl_solver)
endif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

find_library(MKL_CORE_A_PATH mkl_core PATHS ${MKL_LIBRARY_PATH})
find_library(MKL_ARCH_A_PATH ${MKL_ARCH_A} PATHS ${MKL_LIBRARY_PATH})
find_library(MKL_INTEL_A_PATH ${MKL_INTEL_A} PATHS ${MKL_LIBRARY_PATH})
find_library(MKL_SOLVER_A_PATH ${MKL_SOLVER_A} PATHS ${MKL_LIBRARY_PATH})
find_library(MKL_LAPACK_A_PATH mkl_lapack PATHS ${MKL_LIBRARY_PATH})

if(MKL_ARCH_A_PATH)
    if(MKL_INTEL_A_PATH)
#        Version 10+
        set(MKL_LIBRARIES
            ${MKL_INTEL_A_PATH} mkl_intel_thread mkl_core
            guide pthread) 
    else(MKL_INTEL_A_PATH)
#        Version pre-10
        set(MKL_LIBRARIES
            ${MKL_LAPACK_A_PATH} ${MKL_ARCH_A_PATH} guide pthread)
    endif(MKL_INTEL_A_PATH)
endif(MKL_ARCH_A_PATH)

#    MKL 10.2
if(MKL_SOLVER_A_PATH AND NOT MKL_ARCH_A_PATH)
    set(MKL_LIBRARIES
        -L${MKL_LIBRARY_PATH}
        ${MKL_SOLVER_A} ${MKL_INTEL_A} mkl_intel_thread mkl_core
        guide pthread)
endif(MKL_SOLVER_A_PATH AND NOT MKL_ARCH_A_PATH)

if(NOT MKL_FIND_QUIETLY)
    message(STATUS "Found Intel MKL: " ${MKL_PATH})
endif(NOT MKL_FIND_QUIETLY)

else(MKL_FOUND)

if(MKL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Intel MKL")
endif(MKL_FIND_REQUIRED)

endif(MKL_FOUND)

