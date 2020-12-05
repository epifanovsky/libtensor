cmake_minimum_required(VERSION 3.2.0)
project(libtensorlight VERSION 2.9.9 LANGUAGES C CXX)
enable_testing()

#
# Compiler
#
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_SHARED_LIBS "Build shared instead of static libraries" ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(SetupCompiler)

#
# Threading
#
add_definitions(-DPOSIX)
if(NOT UNIX)
    message(FATAL_ERROR "Only unix platforms are supported.")
endif()
include(InspectThreads)
include(InspectStdlib)

#
# BLAS
#
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


#
# Libxm
#
add_definitions(-DWITHOUT_LIBVMM)
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

#
# Pass to subdirectories
#
set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
set(LIBUTIL_SOURCE_DIR ${libtensorlight_SOURCE_DIR})
add_subdirectory(libutil)
add_subdirectory(libtensor)

option(LIBTENSOR_TESTS "Build libtensor tests" ON)
if (LIBTENSOR_TESTS)
    add_subdirectory(tests)
endif()

#
# Installation
#
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

set(PN "libtensorlight")
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Directory to which libraries installed")
install(TARGETS tensorlight util
        EXPORT "${PN}Targets"
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

target_compile_definitions(tensorlight INTERFACE USING_${PN})
target_include_directories(tensorlight INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

get_directory_property(LIBTENSOR_COMPILE_DEFINITIONS_RAW COMPILE_DEFINITIONS)
list(JOIN LIBTENSOR_COMPILE_DEFINITIONS_RAW " -D" LIBTENSOR_COMPILE_DEFINITIONS)
configure_file("${libtensorlight_SOURCE_DIR}/libtensor/libtensorlight.pc.in" "libtensorlight.pc" @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libtensorlight.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
