cmake_minimum_required(VERSION 3.5.0)

##########################################################################
# Project setup

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(libtensorlight VERSION 2.9.9 LANGUAGES C CXX)
enable_testing()

##########################################################################
# Setup compiler flags

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(SetupCompiler)

set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})

##########################################################################
# Threading and stdlib

if(NOT UNIX)
    message(FATAL_ERROR "Only unix platforms are supported.")
endif()
include(InspectThreads)
include(InspectStdlib)

##########################################################################
# Blas

option(BLA_STATIC "Statically link to BLAS and LAPACK" OFF)
if (UNIX AND NOT APPLE)
	set(BLA_VENDOR "OpenBLAS" CACHE STRING "BLAS Vendor to use (see CMake documentation)")
elseif (APPLE)
	set(BLA_VENDOR "Apple" CACHE STRING "BLAS Vendor to use (see CMake documentation)")
endif()
find_package(BLAS REQUIRED)
include_directories(cblas)

if    (BLA_VENDOR STREQUAL "Apple")
elseif(BLA_VENDOR STREQUAL "OpenBLAS")
elseif(BLA_VENDOR STREQUAL "Intel10_64lp_seq" OR BLA_VENDOR STREQUAL "Intel10_64lp")
	# Find the directory with mkl.h
	list(GET BLAS_LIBRARIES 1 BLAS_LIB)
	get_filename_component(BLAS_DIR ${BLAS_LIB} DIRECTORY)
	find_file(MKLINCLUDE mkl.h HINTS "${BLAS_DIR}/../include" "${BLAS_DIR}/../../include")
	get_filename_component(BLAS_INCLUDE_DIR ${MKLINCLUDE} DIRECTORY)
else()
	# TODO If more blas vendors are supported do not forget to add them to
	#      libtensor/linalg/BlasSequential.C and libtensor/CMakeLists.txt
	message(WARNING "BLAS vendor ${BLA_VENDOR} has not been tested with libtensorlight")
endif()
set(BLAS_LAPACK_LIBRARIES ${BLAS_LIBRARIES})

##########################################################################
# Libxm dependencies

option(WITH_LIBXM "Compile libxm support into libtensor" OFF)
option(WITH_MPI   "Compile MPI support into libtensor" OFF)
if (WITH_LIBXM)
    add_definitions(-DWITH_LIBXM)

    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()

    if (WITH_MPI)
        find_package(MPI REQUIRED)
    endif()
else()
    if(WITH_MPI)
        message(FATAL_ERROR "Can't use MPI if WITH_LIBXM is off.")
    endif()
endif()

##########################################################################
# Pass on to subdirectories

add_subdirectory(libutil)
add_subdirectory(libtensor)

option(LIBTENSOR_TESTS "Build libtensor tests" ON)
if (LIBTENSOR_TESTS)
    add_subdirectory(tests)
endif()

##########################################################################
# Installation

# TODO: needs to be optimized, taking almost everything for now
install(DIRECTORY "libtensor/core"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/btod"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/block_tensor"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/gen_block_tensor"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/dense_tensor"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/symmetry"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/expr"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/kernels"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "libtensor/linalg"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY "libutil"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

install(FILES libtensor/libtensor.h libtensor/defs.h libtensor/exception.h
        libtensor/metadata.h libtensor/timings.h libtensor/not_implemented.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libtensor)

install(TARGETS tensorlight
        EXPORT "libtensorlightTargets"
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# Build string of required compile definitions
get_directory_property(LIBTENSOR_COMPILE_DEFINITIONS_RAW COMPILE_DEFINITIONS)
list(JOIN LIBTENSOR_COMPILE_DEFINITIONS_RAW " -D" LIBTENSOR_COMPILE_DEFINITIONS)
if (LIBTENSOR_COMPILE_DEFINITIONS)
    set(LIBTENSOR_COMPILE_DEFINITIONS "-D${LIBTENSOR_COMPILE_DEFINITIONS}")
endif()

# Build the pkg-config file
configure_file("${libtensorlight_SOURCE_DIR}/libtensor/libtensorlight.pc.in" "libtensorlight.pc" @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libtensorlight.pc
        DESTINATION lib/pkgconfig)
