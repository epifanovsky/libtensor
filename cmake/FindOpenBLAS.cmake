#
#    Locates the OpenBLAS library
#
#    Output:
#
#    OPENBLAS_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    OPENBLAS_PATH         - Library home.
#    OPENBLAS_INCLUDE_PATH - Path to the library's header files.
#    OPENBLAS_LIBRARY_PATH - Path to the library's binaries.
#    OPENBLAS_LIBRARIES    - Line for the linker.
#
#    The following locations are searched:
#    1. $QC_EXT_LIBS/OpenBLAS
#    
set(ACML_FOUND FALSE)

set(PATH_OPENBLAS "$ENV{QC_EXT_LIBS}/OpenBLAS")
find_file(CBLAS_H cblas.h PATHS "${PATH_OPENBLAS}/include" NO_DEFAULT_PATH)

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")
find_library(OPENBLAS_A_PATH openblas PATHS "${PATH_OPENBLAS}/lib" NO_DEFAULT_PATH)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES)

if(NOT (CBLAS_H AND OPENBLAS_A_PATH))
    message(SEND_ERROR "OpenBLAS was not found in ${PATH_OPENBLAS}")
endif()

if(OPENBLAS_A_PATH)
    get_filename_component(PATH_OPENBLAS_INCLUDE "${CBLAS_H}" PATH)
    get_filename_component(PATH_OPENBLAS_LIB "${OPENBLAS_A_PATH}" PATH)
    add_library(openblas STATIC IMPORTED)
    set_target_properties(openblas PROPERTIES
        IMPORTED_LOCATION ${OPENBLAS_A_PATH})
    set(OPENBLAS_LIBRARIES openblas)

    set(OPENBLAS_FOUND TRUE)

    if(NOT OPENBLAS_FIND_QUIETLY)
        message(STATUS "Found OpenBLAS: ${PATH_OPENBLAS}")
        foreach(LIB ${OPENBLAS_LIBRARIES})
            set(OPENBLAS_LIBRARIES_FRIENDLY "${OPENBLAS_LIBRARIES_FRIENDLY}${LIB} ")
        endforeach(LIB)
        message(STATUS "OpenBLAS libraries: " ${OPENBLAS_LIBRARIES_FRIENDLY})
        unset(OPENBLAS_LIBRARIES_FRIENDLY)
    endif(NOT OPENBLAS_FIND_QUIETLY)
endif(OPENBLAS_A_PATH)

if(OPENBLAS_FOUND)

#   Check that OpenBLAS compiles and links

set(CMAKE_REQUIRED_LIBRARIES ${OPENBLAS_A_PATH})
check_fortran_source_compiles("
      call xerbla(\"0\",0)
      end
" OPENBLAS_COMPILES)
if(NOT OPENBLAS_COMPILES)
    message(FATAL_ERROR "Compiler and OpenBLAS are not compatible")
endif(NOT OPENBLAS_COMPILES)

#   Set USE_CBLAS definition

add_definitions(-DUSE_CBLAS)
set(USE_CBLAS TRUE)

endif(OPENBLAS_FOUND)

