cmake_minimum_required(VERSION 3.2.0)
project(libtensorlight VERSION 3.0.0 LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

option(CMAKE_BUILD_TYPE "Build type (Release or Debug)" Release)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(SetupCompiler)

if(UNIX)
    add_definitions(-DPOSIX)
endif(UNIX)

# OpenMP
option(ENABLE_OPENMP "Enables OpenMP parallelization" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
	message(WARNING "BLAS vendor ${BLA_VENDOR} has not been tested with adccore")
endif()
set(BLAS_LAPACK_LIBRARIES ${BLAS_LIBRARIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
set(LIBUTIL_SOURCE_DIR ${libtensorlight_SOURCE_DIR})
add_subdirectory(libutil)
add_subdirectory(libtensor)

option(LIBTENSOR_TESTS "Build libtensor tests" ON)
if (LIBTENSOR_TESTS)
    add_subdirectory(tests)
endif()

# TODO: needs to be optimized, taking almost everything for now
set(PN "libtensorlight")
#### Installation ####
option(INSTALL_DEVEL_HEADERS "Install devel headers" ON)
if (INSTALL_DEVEL_HEADERS)
    install(DIRECTORY "libtensor/core"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/btod"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/block_tensor"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/gen_block_tensor"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/dense_tensor"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/symmetry"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/expr"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/kernels"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY "libtensor/linalg"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN}
            FILES_MATCHING PATTERN "*.h")
endif()

install(DIRECTORY "libutil"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

install(FILES libtensor/libtensor.h libtensor/defs.h libtensor/exception.h
        libtensor/version.h libtensor/timings.h libtensor/not_implemented.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PN})

set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Directory to which libraries installed")
install(TARGETS tensorlight util
        EXPORT "${PN}Targets"
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

target_compile_definitions(tensorlight INTERFACE USING_${PN})
target_include_directories(tensorlight INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(CMAKECONFIG_INSTALL_DIR "share/cmake/${PN}")
configure_package_config_file(cmake/${PN}Config.cmake.in
                         "${CMAKE_CURRENT_BINARY_DIR}/${PN}Config.cmake"
                         INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PN}ConfigVersion.cmake
                            VERSION ${${PN}_VERSION}
                            COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PN}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PN}ConfigVersion.cmake
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})

install(EXPORT "${PN}Targets"
        NAMESPACE "${PN}::"
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})

configure_file("${libtensorlight_SOURCE_DIR}/libtensor/libtensorlight.pc.in" "libtensorlight.pc" @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libtensorlight.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
