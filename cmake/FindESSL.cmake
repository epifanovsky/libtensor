#
#    Locates the IBM Engineering and Science Subroutine Library
#
#    Output:
#
#    ESSL_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    ESSL_LIBRARIES    - Line for the linker.
#
set(ESSL_FOUND FALSE)

if(ESSLROOT)
    find_path(ESSL_PATH_ESSLROOT essl.h PATHS ${ESSLROOT}/include NO_DEFAULT_PATH)
endif(ESSLROOT)
if(NOT ESSL_PATH_ESSLROOT)
    find_path(ESSL_PATH_ESSLROOT essl.h PATHS $ENV{ESSLROOT}/include NO_DEFAULT_PATH)
endif(NOT ESSL_PATH_ESSLROOT)

if(ESSL_PATH_ESSLROOT)
    get_filename_component(ESSL_PATH ${ESSL_PATH_ESSLROOT}/.. ABSOLUTE)
    set(ESSL_FOUND TRUE)
endif(ESSL_PATH_ESSLROOT)

if(ESSL_FOUND)

set(ESSL_INCLUDE_PATH ${ESSL_PATH}/include)
set(ESSL_LIBRARY_PATH ${ESSL_PATH}/lib)

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")
if(WITH_OPENMP)
    find_library(ESSL_A_PATH esslsmp PATHS ${ESSL_LIBRARY_PATH} NO_DEFAULT_PATH)
else(WITH_OPENMP)
    find_library(ESSL_A_PATH essl PATHS ${ESSL_LIBRARY_PATH} NO_DEFAULT_PATH)
endif(WITH_OPENMP)
find_library(LAPACK_A_PATH lapack PATHS ${ESSL_LIBRARY_PATH} NO_DEFAULT_PATH)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES)

add_library(essl STATIC IMPORTED)
set_target_properties(essl PROPERTIES IMPORTED_LOCATION ${ESSL_A_PATH})
add_library(lapack STATIC IMPORTED)
set_target_properties(lapack PROPERTIES IMPORTED_LOCATION ${LAPACK_A_PATH})

set(ESSL_LIBRARIES lapack essl)

if(NOT ESSL_FIND_QUIETLY)
    message(STATUS "Found ESSL: " ${ESSL_PATH})
    foreach(LIB ${ESSL_LIBRARIES})
        set(ESSL_LIBRARIES_FRIENDLY "${ESSL_LIBRARIES_FRIENDLY}${LIB} ")
    endforeach(LIB)
    message(STATUS "ESSL libraries: " ${ESSL_LIBRARIES_FRIENDLY})
    unset(ESSL_LIBRARIES_FRIENDLY)
endif(NOT ESSL_FIND_QUIETLY)

else(ESSL_FOUND)

if(ESSL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find ESSL")
endif(ESSL_FIND_REQUIRED)

endif(ESSL_FOUND)

