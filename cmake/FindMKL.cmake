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
    get_filename_component(MKL_PATH ${MKL_PATH_LD}/../../.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_LD)
if(NOT MKL_FOUND AND MKL_PATH_GUESS)
    get_filename_component(MKL_PATH ${MKL_PATH_GUESS}/.. ABSOLUTE)
    set(MKL_FOUND TRUE)
endif(NOT MKL_FOUND AND MKL_PATH_GUESS)

if(MKL_FOUND)

#if(APPLE)
#    set(MKL_INCLUDE_PATH ${MKL_PATH}/Headers)
#    set(MKL_LIB_PATH ${MKL_PATH}/Versions/Current/lib)
#else(APPLE)
    set(MKL_INCLUDE_PATH ${MKL_PATH}/include)
    set(MKL_LIB_PATH ${MKL_PATH}/lib)
#endif(APPLE)


#   32-bit or 64-bit?

if(APPLE OR (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64"))
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

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")

#   Determine the version of MKL

if(MKL_32)
    find_library(MKL_ARCH_PATH mkl_ia32 PATHS ${MKL_LIBRARY_PATH})
    find_library(MKL_LAPACK_PATH mkl_lapack PATHS ${MKL_LIBRARY_PATH})
    find_library(MKL_INTEL_PATH mkl_intel PATHS ${MKL_LIBRARY_PATH})
    if(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver PATHS ${MKL_LIBRARY_PATH})
    else(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver_sequential
            PATHS ${MKL_LIBRARY_PATH})
    endif(WITH_OPENMP)
else(MKL_32)
    find_library(MKL_ARCH_PATH mkl_em64t PATHS ${MKL_LIBRARY_PATH})
    find_library(MKL_LAPACK_PATH mkl_lapack PATHS ${MKL_LIBRARY_PATH})
    find_library(MKL_INTEL_PATH mkl_intel_lp64 PATHS ${MKL_LIBRARY_PATH})
    if(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver_lp64 PATHS ${MKL_LIBRARY_PATH})
    else(WITH_OPENMP)
        find_library(MKL_SOLVER_PATH mkl_solver_lp64_sequential
            PATHS ${MKL_LIBRARY_PATH})
    endif(WITH_OPENMP)
endif(MKL_32)
if(MKL_ARCH_PATH AND MKL_LAPACK_PATH)
    if(MKL_INTEL_PATH)
        set(MKL_VERSION "10.1")
    else(MKL_INTEL_PATH)
        set(MKL_VERSION "9.1")
    endif(MKL_INTEL_PATH)
endif(MKL_ARCH_PATH AND MKL_LAPACK_PATH)
if(NOT MKL_ARCH_PATH AND MKL_SOLVER_PATH)
    if(MKL_2011)
        set(MKL_VERSION "10.3")
    else(MKL_2011)
        set(MKL_VERSION "10.2")
    endif(MKL_2011)
endif(NOT MKL_ARCH_PATH AND MKL_SOLVER_PATH)
if(MKL_VERSION VERSION_GREATER "10.0")
    find_library(MKL_CORE_PATH mkl_core PATHS ${MKL_LIBRARY_PATH})
    if(WITH_OPENMP)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            find_library(MKL_THREAD_PATH mkl_gnu_thread
                PATHS ${MKL_LIBRARY_PATH})
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            find_library(MKL_THREAD_PATH mkl_intel_thread
                PATHS ${MKL_LIBRARY_PATH})
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
            find_library(MKL_THREAD_PATH mkl_pgi_thread
                PATHS ${MKL_LIBRARY_PATH})
        endif()
    else(WITH_OPENMP)
        find_library(MKL_THREAD_PATH mkl_sequential PATHS ${MKL_LIBRARY_PATH})
    endif(WITH_OPENMP)
    if(WITH_SCALAPACK)
        if(WITH_MPI STREQUAL "openmpi")
            if(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_core
                    PATHS ${MKL_LIBRARY_PATH})
                find_library(MKL_BLACS_PATH mkl_blacs_openmpi
                    PATHS ${MKL_LIBRARY_PATH})
            else(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_lp64
                    PATHS ${MKL_LIBRARY_PATH})
                find_library(MKL_BLACS_PATH mkl_blacs_openmpi_lp64
                    PATHS ${MKL_LIBRARY_PATH})
            endif(MKL_32)
        else()
            if(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_core
                    PATHS ${MKL_LIBRARY_PATH})
                find_library(MKL_BLACS_PATH mkl_blacs PATHS ${MKL_LIBRARY_PATH})
            else(MKL_32)
                find_library(MKL_SCALAPACK_PATH mkl_scalapack_lp64
                    PATHS ${MKL_LIBRARY_PATH})
                find_library(MKL_BLACS_PATH mkl_blacs_lp64
                    PATHS ${MKL_LIBRARY_PATH})
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

if(MKL_VERSION VERSION_EQUAL "9.1")
    add_library(mkl_lapack STATIC IMPORTED)
    set_target_properties(mkl_lapack PROPERTIES
        IMPORTED_LOCATION ${MKL_LAPACK_PATH})
    add_library(mkl_arch STATIC IMPORTED)
    set_target_properties(mkl_arch PROPERTIES
        IMPORTED_LOCATION ${MKL_ARCH_PATH})
    set(MKL_LIBRARIES mkl_lapack mkl_arch)
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
    else(APPLE)
        set(MKL_LIBRARIES mkl_solver -Wl,--start-group mkl_intel mkl_thread
            mkl_core -Wl,--end-group) 
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
    else(MKL_SCALAPACK)
        if(APPLE)
            set(MKL_LIBRARIES mkl_intel mkl_thread mkl_core)
        else(APPLE)
            set(MKL_LIBRARIES mkl_solver -Wl,--start-group mkl_intel mkl_thread
                mkl_core -Wl,--end-group)
        endif(APPLE)
    endif(MKL_SCALAPACK)
endif(MKL_VERSION VERSION_EQUAL "10.2")

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
    else(MKL_SCALAPACK)
        if(APPLE)
            set(MKL_LIBRARIES mkl_intel mkl_thread mkl_core)
        else(APPLE)
            set(MKL_LIBRARIES -Wl,--start-group mkl_intel mkl_thread mkl_core
                -Wl,--end-group)
        endif(APPLE)
    endif(MKL_SCALAPACK)
endif(MKL_VERSION VERSION_EQUAL "10.3")

if(WITH_SCALAPACK AND (NOT MKL_SCALAPACK))
    find_library(ScaLAPACK)
    set(MKL_LIBRARIES ${SCALAPACK_LIBRARIES} ${MKL_LIBRARIES})
else()
    set(SCALAPACK_FOUND TRUE)
endif()

#   Done!

if(NOT MKL_FIND_QUIETLY)
    message(STATUS "Found Intel MKL " ${MKL_VERSION} ": " ${MKL_PATH})
    foreach(LIB ${MKL_LIBRARIES})
        set(MKL_LIBRARIES_FRIENDLY "${MKL_LIBRARIES_FRIENDLY}${LIB} ")
    endforeach(LIB)
    message(STATUS "MKL libraries: " ${MKL_LIBRARIES_FRIENDLY})
    unset(MKL_LIBRARIES_FRIENDLY)
endif(NOT MKL_FIND_QUIETLY)

else(MKL_FOUND)

if(MKL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Intel MKL")
endif(MKL_FIND_REQUIRED)

endif(MKL_FOUND)

