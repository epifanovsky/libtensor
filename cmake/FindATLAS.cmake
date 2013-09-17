set(ATLAS_FOUND FALSE)

#
#   1. Inspect $QC_EXT_LIBS/atlas
#
set(PATH_ATLAS "$ENV{QC_EXT_LIBS}/atlas")
find_file(CBLAS_H cblas.h PATHS "${PATH_ATLAS}/include" NO_DEFAULT_PATH)
find_file(ATLAS_BUILDINFO_H atlas/atlas_buildinfo.h
    PATHS "${PATH_ATLAS}/include" NO_DEFAULT_PATH)

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")
find_library(ATLAS_A_PATH atlas PATHS "${PATH_ATLAS}/lib" NO_DEFAULT_PATH)
find_library(CBLAS_A_PATH cblas PATHS "${PATH_ATLAS}/lib" NO_DEFAULT_PATH)
find_library(F77BLAS_A_PATH f77blas PATHS "${PATH_ATLAS}/lib" NO_DEFAULT_PATH)
find_library(LAPACK_A_PATH lapack PATHS "${PATH_ATLAS}/lib" NO_DEFAULT_PATH)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES)

if(NOT (CBLAS_H AND ATLAS_BUILDINFO_H AND ATLAS_A_PATH))
    message(SEND_ERROR "Proper ATLAS was not found in ${PATH_ATLAS}")
endif()

#
#   2. Inspect standard locations
#

#
#   3. Set output variables
#
if(ATLAS_A_PATH)
    get_filename_component(PATH_ATLAS_INCLUDE "${CBLAS_H}" PATH)
    get_filename_component(PATH_ATLAS_LIB "${ATLAS_A_PATH}" PATH)
    add_library(atlas_blas STATIC IMPORTED)
    set_target_properties(atlas_blas PROPERTIES
        IMPORTED_LOCATION ${ATLAS_A_PATH})
    add_library(atlas_cblas STATIC IMPORTED)
    set_target_properties(atlas_cblas PROPERTIES
        IMPORTED_LOCATION ${CBLAS_A_PATH})
    add_library(atlas_f77blas STATIC IMPORTED)
    set_target_properties(atlas_f77blas PROPERTIES
        IMPORTED_LOCATION ${F77BLAS_A_PATH})
    add_library(atlas_lapack STATIC IMPORTED)
    set_target_properties(atlas_lapack PROPERTIES
        IMPORTED_LOCATION ${LAPACK_A_PATH})
    set(ATLAS_LIBRARIES atlas_lapack atlas_f77blas atlas_cblas atlas_blas)

    set(ATLAS_FOUND TRUE)

    if(NOT ATLAS_FIND_QUIETLY)
        message(STATUS "Found ATLAS: ${PATH_ATLAS}")
        foreach(LIB ${ATLAS_LIBRARIES})
            set(ATLAS_LIBRARIES_FRIENDLY "${ATLAS_LIBRARIES_FRIENDLY}${LIB} ")
        endforeach(LIB)
        message(STATUS "ATLAS libraries: " ${ATLAS_LIBRARIES_FRIENDLY})
        unset(ATLAS_LIBRARIES_FRIENDLY)
    endif(NOT ATLAS_FIND_QUIETLY)
endif(ATLAS_A_PATH)

