#
#   Locates the Intel Math Kernel Library
#
#   Input:
#
#   MKLROOT (CMake or environment) - Preferred MKL path.
#   STATIC_LINK                    - Static (true)/dynamic (false) linking.
#   WITH_OPENMP                    - Enables OpenMP parallel MKL libraries.
#   WITH_SCALAPACK                 - Enables ScaLAPACK/BLACS libraries.
#   WITH_MPI                       - Specifies MPI library.
#                                    (openmpi, mpich, ...)
#
#   Output:
#
#   MKL_FOUND - TRUE/FALSE - Whether the library has been found.
#       If FALSE, all other output variables are not defined.
#
#   MKL_PATH         - Library home.
#   MKL_INCLUDE_PATH - Path to the library's header files.
#   MKL_LIBRARY_PATH - Path to the library's binaries.
#   MKL_LIBRARIES    - Line for the linker.
#
#   The following locations are searched:
#   1. CMake MKLROOT
#   2. Environment MKLROOT
#   3. Intel Compiler directory
#   4. LD_LIBRARY_PATH/DYLD_LIBRARY_PATH/PATH
#   5. Default MKL installation directories
#	
set(MKL_FOUND FALSE)

#
#   Set up search locations
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

if(NOT MKLROOT)
    set(MKLROOT $ENV{MKLROOT})
endif(NOT MKLROOT)
if(MKLROOT)
    find_path(MKL_PATH_MKLROOT mkl.h PATHS ${MKLROOT}/include NO_DEFAULT_PATH)
endif(MKLROOT)

if(CMAKE_C_COMPILER_ID STREQUAL "Intel")
    get_filename_component(MKL_PATH_ICC ${CMAKE_C_COMPILER}/../../.. ABSOLUTE)
    find_path(MKL_PATH_ICC1 mkl.h PATHS ${MKL_PATH_ICC}/mkl/include
        NO_DEFAULT_PATH)
    unset(MKL_PATH_ICC)
    if(MKL_PATH_ICC1)
        get_filename_component(MKL_PATH_ICC ${MKL_PATH_ICC1}/.. ABSOLUTE)
    endif(MKL_PATH_ICC1)
    unset(MKL_PATH_ICC1)
endif(CMAKE_C_COMPILER_ID STREQUAL "Intel")

find_library(MKL_PATH_LD NAMES mkl mkl_core PATHS ENV ${LD_LIBRARY_PATH_NAME})

find_path(MKL_PATH_GUESS mkl.h
    PATHS
    $ENV{HOME}/intel/composerxe*/mkl/include
    /opt/intel/composerxe*/mkl/include
    $ENV{HOME}/intel/Compiler/*/mkl/include
    /opt/intel/Compiler/*/mkl/include
    $ENV{HOME}/intel/mkl/*/include
    /opt/intel/mkl/*/include)

if(MKL_PATH_MKLROOT)
    get_filename_component(MKL_PATH ${MKL_PATH_MKLROOT}/.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(MKL_PATH_MKLROOT)
if(NOT MKL_FOUND AND MKL_PATH_ICC)
    set(MKL_PATH ${MKL_PATH_ICC})
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_ICC)
if(NOT MKL_FOUND AND MKL_PATH_LD)
    get_filename_component(MKL_PATH ${MKL_PATH_LD} PATH)
    get_filename_component(MKL_PATH ${MKL_PATH}/.. ABSOLUTE)
    if(NOT EXISTS ${MKL_PATH}/lib)
        get_filename_component(MKL_PATH ${MKL_PATH}/.. ABSOLUTE)
    endif()
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_LD)
if(NOT MKL_FOUND AND MKL_PATH_GUESS)
    get_filename_component(MKL_PATH ${MKL_PATH_GUESS}/.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_GUESS)

if(MKL_FOUND)

if(APPLE)
    if(IS_DIRECTORY "${MKL_PATH}/include")
        set(MKL_INCLUDE_PATH ${MKL_PATH}/include)
    else()
        set(MKL_INCLUDE_PATH ${MKL_PATH}/Headers)
    endif()
    if(IS_DIRECTORY "${MKL_PATH}/lib")
        set(MKL_LIB_PATH ${MKL_PATH}/lib)
    else()
        set(MKL_LIB_PATH ${MKL_PATH}/Versions/Current/lib)
    endif()
else(APPLE)
    set(MKL_INCLUDE_PATH ${MKL_PATH}/include)
    set(MKL_LIB_PATH ${MKL_PATH}/lib)
endif(APPLE)


#   Determine directory structure, infer some version info
#
#   MKL 11.0 (Composer XE 2013)
#    - Linux: mkl/lib/(intel64,mic)
#    - Mac: mkl/lib (no subdirectories)
#   MKL 10.3 (Composer XE 2011)
#    - Linux: mkl/lib/(ia32,intel64)
#    - Mac: mkl/lib/(ia32,intel64,universal)
#   MKL 10.2 and earlier
#    - Linux: mkl/lib/(32,em64t)
#    - Mac: mkl/lib/(32,em64t,universal)

if(APPLE)
    set(MKL_32 0)
    if(EXISTS ${MKL_LIB_PATH}/intel64)
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/intel64)
        set(MKL_2011 TRUE)
    elseif(EXISTS ${MKL_LIB_PATH}/em64t)
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/em64t)
        set(MKL_2011 FALSE)
    else()
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH})
        set(MKL_2013 TRUE)
    endif()
    if(EXISTS ${MKL_LIB_PATH}/universal)
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/universal)
    endif()
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(MKL_32 0)
    if(EXISTS ${MKL_LIB_PATH}/intel64)
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/intel64)
        set(MKL_2011 TRUE)
    else()
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/em64t)
        set(MKL_2011 FALSE)
    endif()
else()
    set(MKL_32 1)
    if(EXISTS ${MKL_LIB_PATH}/ia32)
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/ia32)
        set(MKL_2011 TRUE)
    else()
        set(MKL_LIBRARY_PATH ${MKL_LIB_PATH}/32)
        set(MKL_2011 FALSE)
    endif()
endif()
if(EXISTS ${MKL_LIB_PATH}/mic)
    set(MKL_MIC 1)
endif()

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")

#   Determine the version of MKL

if(MKL_32)
    find_library(MKL_ARCH_PATH mkl_ia32
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    find_library(MKL_LAPACK_PATH mkl_lapack
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    find_library(MKL_INTEL_PATH mkl_intel
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    if(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver
            PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    else(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver_sequential
            PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    endif(WITH_OPENMP)
    find_library(MKL_LAPACK95_PATH mkl_lapack95
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    find_library(MKL_CORE_PATH mkl_core
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
else(MKL_32)
    find_library(MKL_ARCH_PATH mkl_em64t
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    find_library(MKL_LAPACK_PATH mkl_lapack
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    find_library(MKL_INTEL_PATH mkl_intel_lp64
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    if(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver_lp64
            PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    else(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver_lp64_sequential
            PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    endif(WITH_OPENMP)
    find_library(MKL_LAPACK95_PATH mkl_lapack95_lp64
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    find_library(MKL_CORE_PATH mkl_core
        PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
endif(MKL_32)
if(MKL_ARCH_PATH AND MKL_LAPACK_PATH)
    if(MKL_INTEL_PATH)
        set(MKL_VERSION "10.1")
    else(MKL_INTEL_PATH)
        set(MKL_VERSION "9.1")
    endif(MKL_INTEL_PATH)
endif(MKL_ARCH_PATH AND MKL_LAPACK_PATH)
if(NOT MKL_ARCH_PATH AND MKL_CORE_PATH AND MKL_SOLVER_PATH)
    if(MKL_2011)
        set(MKL_VERSION "10.3")
    else(MKL_2011)
        set(MKL_VERSION "10.2")
    endif(MKL_2011)
endif(NOT MKL_ARCH_PATH AND MKL_CORE_PATH AND MKL_SOLVER_PATH)
if((NOT MKL_ARCH_PATH) AND MKL_CORE_PATH AND (NOT MKL_SOLVER_PATH)
        AND MKL_LAPACK95_PATH)
    set(MKL_VERSION "11.0")
endif()
if(NOT MKL_VERSION)
    message(STATUS "WARNING: Failed to determine the version of MKL")
endif(NOT MKL_VERSION)
if(MKL_VERSION VERSION_GREATER "10.0")
    find_library(INTEL_IOMP5_PATH iomp5 PATHS ENV ${LD_LIBRARY_PATH_NAME})
    if(WITH_OPENMP)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(APPLE)
                find_library(MKL_THREAD_PATH mkl_intel_thread
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            else(APPLE)
                find_library(MKL_THREAD_PATH mkl_gnu_thread
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            endif(APPLE)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            find_library(MKL_THREAD_PATH mkl_intel_thread
                PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
            find_library(MKL_THREAD_PATH mkl_pgi_thread
                PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
        endif()
    else(WITH_OPENMP)
        find_library(MKL_THREAD_PATH mkl_sequential
            PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
    endif(WITH_OPENMP)
    if(WITH_SCALAPACK)
        if(WITH_MPI STREQUAL "openmpi")
            if(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_core
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
                find_library(MKL_BLACS_PATH mkl_blacs_openmpi
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            else(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_lp64
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
                find_library(MKL_BLACS_PATH mkl_blacs_openmpi_lp64
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            endif(MKL_32)
        elseif(WITH_MPI STREQUAL "intelmpi")
            if(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_core
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
                find_library(MKL_BLACS_PATH mkl_blacs_intelmpi
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            else(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_lp64
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
                find_library(MKL_BLACS_PATH mkl_blacs_intelmpi_lp64
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            endif(MKL_32)
        else()
            if(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_core
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
                find_library(MKL_BLACS_PATH mkl_blacs
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            else(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_lp64
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
                find_library(MKL_BLACS_PATH mkl_blacs_lp64
                    PATHS ${MKL_LIBRARY_PATH} NO_DEFAULT_PATH)
            endif(MKL_32)
        endif()
        if(MKL_SCALAPACK_PATH)
            set(MKL_SCALAPACK TRUE)
        endif(MKL_SCALAPACK_PATH)
    endif(WITH_SCALAPACK)
endif(MKL_VERSION VERSION_GREATER "10.0")

set(CMAKE_FIND_LIBRARY_SUFFIXES ${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES)

#   Compose the linker line

if(INTEL_IOMP5_PATH)
    add_library(iomp5 STATIC IMPORTED)
    set_target_properties(iomp5 PROPERTIES
        IMPORTED_LOCATION ${INTEL_IOMP5_PATH})
endif(INTEL_IOMP5_PATH)

if(MKL_VERSION VERSION_EQUAL "9.1")
    add_library(mkl_lapack STATIC IMPORTED)
    set_target_properties(mkl_lapack PROPERTIES
        IMPORTED_LOCATION ${MKL_LAPACK_PATH})
    add_library(mkl_arch STATIC IMPORTED)
    set_target_properties(mkl_arch PROPERTIES
        IMPORTED_LOCATION ${MKL_ARCH_PATH})
    set(MKL_LIBRARIES mkl_lapack mkl_arch)
    set(MKL_LIBRARIES_EXPLICIT ${MKL_LAPACK_PATH} ${MKL_ARCH_PATH})
endif(MKL_VERSION VERSION_EQUAL "9.1")

if(MKL_VERSION VERSION_EQUAL "10.1")
    add_library(mkl_solver STATIC IMPORTED)
    set_target_properties(mkl_solver PROPERTIES
        IMPORTED_LOCATION ${MKL_SOLVER_PATH})
    add_library(mkl_intel STATIC IMPORTED)
    set_target_properties(mkl_intel PROPERTIES
        IMPORTED_LOCATION ${MKL_INTEL_PATH})
    add_library(mkl_thread STATIC IMPORTED)
    set_target_properties(mkl_thread PROPERTIES
        IMPORTED_LOCATION ${MKL_THREAD_PATH})
    add_library(mkl_core STATIC IMPORTED)
    set_target_properties(mkl_core PROPERTIES
        IMPORTED_LOCATION ${MKL_CORE_PATH})
    if(APPLE)
        set(MKL_LIBRARIES mkl_intel mkl_thread mkl_core) 
        set(MKL_LIBRARIES_EXPLICIT ${MKL_INTEL_PATH} ${MKL_THREAD_PATH}
            ${MKL_CORE_PATH})
        if(WITH_OPENMP)
            set(MKL_LIBRARIES ${MKL_LIBRARIES} iomp5)
            set(MKL_LIBRARIES_EXPLICIT
                ${MKL_LIBRARIES_EXPLICIT} ${INTEL_IOMP5_PATH})
        endif(WITH_OPENMP)
    else(APPLE)
        set(MKL_LIBRARIES mkl_solver -Wl,--start-group mkl_intel mkl_thread
            mkl_core -Wl,--end-group) 
        set(MKL_LIBRARIES_EXPLICIT ${MKL_SOLVER_PATH} ${MKL_INTEL_PATH}
            ${MKL_THREAD_PATH} ${MKL_CORE_PATH})
    endif(APPLE)
endif(MKL_VERSION VERSION_EQUAL "10.1")

if(MKL_VERSION VERSION_EQUAL "10.2")
    add_library(mkl_solver STATIC IMPORTED)
    set_target_properties(mkl_solver PROPERTIES
        IMPORTED_LOCATION ${MKL_SOLVER_PATH})
    add_library(mkl_intel STATIC IMPORTED)
    set_target_properties(mkl_intel PROPERTIES
        IMPORTED_LOCATION ${MKL_INTEL_PATH})
    add_library(mkl_thread STATIC IMPORTED)
    set_target_properties(mkl_thread PROPERTIES
        IMPORTED_LOCATION ${MKL_THREAD_PATH})
    add_library(mkl_core STATIC IMPORTED)
    set_target_properties(mkl_core PROPERTIES
        IMPORTED_LOCATION ${MKL_CORE_PATH})
    if(MKL_SCALAPACK)
        add_library(mkl_scalapack STATIC IMPORTED)
        set_target_properties(mkl_scalapack PROPERTIES
            IMPORTED_LOCATION ${MKL_SCALAPACK_PATH})
        add_library(mkl_blacs STATIC IMPORTED)
        set_target_properties(mkl_blacs PROPERTIES
            IMPORTED_LOCATION ${MKL_BLACS_PATH})
        set(MKL_LIBRARIES mkl_scalapack mkl_solver -Wl,--start-group mkl_intel
            mkl_thread mkl_core mkl_blacs -Wl,--end-group)
        set(MKL_LIBRARIES_EXPLICIT ${MKL_SCALAPACK_PATH} ${MKL_SOLVER_PATH}
            ${MKL_INTEL_PATH} ${MKL_THREAD_PATH} ${MKL_CORE_PATH}
            ${MKL_BLACS_PATH})
    else(MKL_SCALAPACK)
        if(APPLE)
            set(MKL_LIBRARIES mkl_intel mkl_thread mkl_core)
            set(MKL_LIBRARIES_EXPLICIT ${MKL_INTEL_PATH} ${MKL_THREAD_PATH}
                ${MKL_CORE_PATH})
            if(WITH_OPENMP)
                set(MKL_LIBRARIES ${MKL_LIBRARIES} iomp5)
                set(MKL_LIBRARIES_EXPLICIT
                    ${MKL_LIBRARIES_EXPLICIT} ${INTEL_IOMP5_PATH})
            endif(WITH_OPENMP)
        else(APPLE)
            set(MKL_LIBRARIES mkl_solver -Wl,--start-group mkl_intel mkl_thread
                mkl_core -Wl,--end-group)
            set(MKL_LIBRARIES_EXPLICIT ${MKL_SOLVER_PATH} -Wl,--start-group
                ${MKL_INTEL_PATH} ${MKL_THREAD_PATH} ${MKL_CORE_PATH}
                -Wl,--end-group)
        endif(APPLE)
    endif(MKL_SCALAPACK)
endif(MKL_VERSION VERSION_EQUAL "10.2")

#  10.3
if(MKL_VERSION VERSION_EQUAL "10.3")
    add_library(mkl_intel STATIC IMPORTED)
    set_target_properties(mkl_intel PROPERTIES
        IMPORTED_LOCATION ${MKL_INTEL_PATH})
    add_library(mkl_thread STATIC IMPORTED)
    set_target_properties(mkl_thread PROPERTIES
        IMPORTED_LOCATION ${MKL_THREAD_PATH})
    add_library(mkl_core STATIC IMPORTED)
    set_target_properties(mkl_core PROPERTIES
        IMPORTED_LOCATION ${MKL_CORE_PATH})
    if(MKL_SCALAPACK)
        add_library(mkl_scalapack STATIC IMPORTED)
        set_target_properties(mkl_scalapack PROPERTIES
            IMPORTED_LOCATION ${MKL_SCALAPACK_PATH})
        add_library(mkl_blacs STATIC IMPORTED)
        set_target_properties(mkl_blacs PROPERTIES
            IMPORTED_LOCATION ${MKL_BLACS_PATH})
        set(MKL_LIBRARIES mkl_scalapack -Wl,--start-group mkl_intel mkl_thread
            mkl_core mkl_blacs -Wl,--end-group) 
        set(MKL_LIBRARIES_EXPLICIT ${MKL_SCALAPACK_PATH} ${MKL_INTEL_PATH}
            ${MKL_THREAD_PATH} ${MKL_CORE_PATH} ${MKL_BLACS_PATH})
    else(MKL_SCALAPACK)
        if(APPLE)
            set(MKL_LIBRARIES mkl_intel mkl_thread mkl_core)
            set(MKL_LIBRARIES_EXPLICIT ${MKL_INTEL_PATH} ${MKL_THREAD_PATH}
                ${MKL_CORE_PATH})
            if(WITH_OPENMP)
                set(MKL_LIBRARIES ${MKL_LIBRARIES} iomp5)
                set(MKL_LIBRARIES_EXPLICIT
                    ${MKL_LIBRARIES_EXPLICIT} ${INTEL_IOMP5_PATH})
            endif(WITH_OPENMP)
        else(APPLE)
            set(MKL_LIBRARIES -Wl,--start-group mkl_intel mkl_thread mkl_core
                -Wl,--end-group)
            set(MKL_LIBRARIES_EXPLICIT -Wl,--start-group ${MKL_INTEL_PATH}
                ${MKL_THREAD_PATH} ${MKL_CORE_PATH} -Wl,--end-group)
        endif(APPLE)
    endif(MKL_SCALAPACK)
endif(MKL_VERSION VERSION_EQUAL "10.3")

#  11.0
if(MKL_VERSION VERSION_EQUAL "11.0")
    add_library(mkl_intel STATIC IMPORTED)
    set_target_properties(mkl_intel PROPERTIES
        IMPORTED_LOCATION ${MKL_INTEL_PATH})
    add_library(mkl_thread STATIC IMPORTED)
    set_target_properties(mkl_thread PROPERTIES
        IMPORTED_LOCATION ${MKL_THREAD_PATH})
    add_library(mkl_core STATIC IMPORTED)
    set_target_properties(mkl_core PROPERTIES
        IMPORTED_LOCATION ${MKL_CORE_PATH})
    if(MKL_SCALAPACK)
        add_library(mkl_scalapack STATIC IMPORTED)
        set_target_properties(mkl_scalapack PROPERTIES
            IMPORTED_LOCATION ${MKL_SCALAPACK_PATH})
        add_library(mkl_blacs STATIC IMPORTED)
        set_target_properties(mkl_blacs PROPERTIES
            IMPORTED_LOCATION ${MKL_BLACS_PATH})
        set(MKL_LIBRARIES mkl_scalapack -Wl,--start-group mkl_intel mkl_thread
            mkl_core mkl_blacs -Wl,--end-group dl) 
        set(MKL_LIBRARIES_EXPLICIT ${MKL_SCALAPACK_PATH} ${MKL_INTEL_PATH}
            ${MKL_THREAD_PATH} ${MKL_CORE_PATH} ${MKL_BLACS_PATH} dl)
    else(MKL_SCALAPACK)
        if(APPLE)
            set(MKL_LIBRARIES mkl_intel mkl_thread mkl_core)
            set(MKL_LIBRARIES_EXPLICIT ${MKL_INTEL_PATH} ${MKL_THREAD_PATH}
                ${MKL_CORE_PATH})
            if(WITH_OPENMP)
                set(MKL_LIBRARIES ${MKL_LIBRARIES} iomp5)
                set(MKL_LIBRARIES_EXPLICIT
                    ${MKL_LIBRARIES_EXPLICIT} ${INTEL_IOMP5_PATH})
            endif(WITH_OPENMP)
        else(APPLE)
            set(MKL_LIBRARIES -Wl,--start-group mkl_intel mkl_thread mkl_core
                -Wl,--end-group pthread dl)
            set(MKL_LIBRARIES_EXPLICIT -Wl,--start-group ${MKL_INTEL_PATH}
                ${MKL_THREAD_PATH} ${MKL_CORE_PATH} -Wl,--end-group pthread dl)
        endif(APPLE)
    endif(MKL_SCALAPACK)
endif(MKL_VERSION VERSION_EQUAL "11.0")

if(WITH_SCALAPACK AND (NOT MKL_SCALAPACK))
    find_package(ScaLAPACK)
    set(MKL_LIBRARIES ${SCALAPACK_LIBRARIES} ${MKL_LIBRARIES})
else()
    set(SCALAPACK_FOUND TRUE)
endif()

#   Inform the user

if(NOT MKL_FIND_QUIETLY)
    message(STATUS "Found Intel MKL ${MKL_VERSION}: ${MKL_PATH}")
    foreach(LIB ${MKL_LIBRARIES})
        set(MKL_LIBRARIES_FRIENDLY "${MKL_LIBRARIES_FRIENDLY}${LIB} ")
    endforeach(LIB)
    message(STATUS "MKL libraries: " ${MKL_LIBRARIES_FRIENDLY})
    unset(MKL_LIBRARIES_FRIENDLY)
endif(NOT MKL_FIND_QUIETLY)

#   Set USE_MKL definition

add_definitions(-DUSE_MKL)
set(USE_MKL TRUE)

#   Test features

set(CMAKE_REQUIRED_INCLUDES ${MKL_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${MKL_LIBRARIES_EXPLICIT})

check_cxx_source_compiles("
#include <mkl.h>
int main() {
    int nx = 10, incx = 1, incy = 1;
    double x[10], y[10];
    for(int i = 0; i < 10; i++) x[i] = double(i);
    dcopy(&nx, x, &incx, y, &incy);
    return 0;
}
" MKL_COMPILES)
if(NOT MKL_COMPILES)
    message(FATAL_ERROR "Unable to compile a simple MKL program")
endif(NOT MKL_COMPILES)

check_cxx_source_compiles("
#include <mkl.h>
int main() {
    double a[10], b[20], c[10]; vdAdd(10, a, b, c);
    return 0;
}
" HAVE_MKL_VML)
if(HAVE_MKL_VML)
    add_definitions(-DHAVE_MKL_VML)
endif(HAVE_MKL_VML)

check_cxx_source_compiles("
#include <mkl.h>
int main() {
  double a[10];
  VSLStreamStatePtr stream;
  vslNewStream(&stream, VSL_BRNG_MT19937, 777);
  vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 10, a, 0.0, 1.0);
  vslDeleteStream(&stream);
  return 0;
}
" HAVE_MKL_VSL)
if(HAVE_MKL_VSL)
    add_definitions(-DHAVE_MKL_VSL)
endif(HAVE_MKL_VSL)

check_cxx_source_compiles("
#include <mkl.h>
#include <mkl_trans.h>
int main() {
    double a[9], b[9];
    mkl_domatcopy('C', 'N', 3, 3, 1.0, a, 3, b, 3);
    return 0;
}
" HAVE_MKL_DOMATCOPY)
if(HAVE_MKL_DOMATCOPY)
    add_definitions(-DHAVE_MKL_DOMATCOPY)
endif(HAVE_MKL_DOMATCOPY)

#   Done!

else(MKL_FOUND)

if(MKL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Intel MKL")
endif(MKL_FIND_REQUIRED)

endif(MKL_FOUND)

