#
#    Locates the AMD Core Math Library
#
#    Output:
#
#    ACML_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    ACML_PATH         - Library home.
#    ACML_INCLUDE_PATH - Path to the library's header files.
#    ACML_LIBRARY_PATH - Path to the library's binaries.
#    ACML_LIBRARIES    - Line for the linker.
#
#    The following locations are searched:
#    1. CMake ACMLROOT
#    2. Environment ACMLROOT
#    4. Default ACML installation directories
#    
set(ACML_FOUND FALSE)

if(ACMLROOT)
    find_path(ACML_PATH_ACMLROOT acml.h PATHS ${ACMLROOT}/include)
endif(ACMLROOT)
if(NOT ACML_PATH_ACMLROOT)
    find_path(ACML_PATH_ACMLROOT acml.h PATHS $ENV{ACMLROOT}/include)
endif(NOT ACML_PATH_ACMLROOT)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(ACML_SUBDIR "gfortran64")
    else()
        set(ACML_SUBDIR "gfortran")
    endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(ACML_SUBDIR "ifort64")
    else()
        set(ACML_SUBDIR "ifort")
    endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(ACML_SUBDIR "pgi64")
    else()
        set(ACML_SUBDIR "pgi")
    endif()
else()
    set(ACML_SUBDIR "unknown")
endif()

find_path(ACML_PATH_GUESS include/acml.h
    PATHS
    "/opt/acml5.3.0/${ACML_SUBDIR}"
    "$ENV{HOME}/opt/acml5.3.0/${ACML_SUBDIR}"
    "/opt/acml5.2.0/${ACML_SUBDIR}"
    "$ENV{HOME}/opt/acml5.2.0/${ACML_SUBDIR}"
    "/opt/acml5.1.0/${ACML_SUBDIR}"
    "$ENV{HOME}/opt/acml5.1.0/${ACML_SUBDIR}"
    "/opt/acml4.4.0/${ACML_SUBDIR}"
    "$ENV{HOME}/opt/acml4.4.0/${ACML_SUBDIR}")

if(ACML_PATH_ACMLROOT)
    get_filename_component(ACML_PATH ${ACML_PATH_ACMLROOT}/.. ABSOLUTE)
    set(ACML_FOUND TRUE)
endif(ACML_PATH_ACMLROOT)
if(NOT ACML_FOUND AND ACML_PATH_GUESS)
    get_filename_component(ACML_PATH ${ACML_PATH_GUESS} ABSOLUTE)
    set(ACML_FOUND TRUE)
endif(NOT ACML_FOUND AND ACML_PATH_GUESS)

if(ACML_FOUND)

set(ACML_INCLUDE_PATH ${ACML_PATH}/include)
set(ACML_LIBRARY_PATH ${ACML_PATH}/lib)

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")
find_library(ACML_A_PATH acml PATHS ${ACML_LIBRARY_PATH} NO_DEFAULT_PATH)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES)

add_library(acml STATIC IMPORTED)
set_target_properties(acml PROPERTIES IMPORTED_LOCATION ${ACML_A_PATH})

#set(ACML_LIBRARIES -L${ACML_LIBRARY_PATH} acml fortran ffio)
set(ACML_LIBRARIES acml)

if(NOT ACML_FIND_QUIETLY)
    message(STATUS "Found ACML: " ${ACML_PATH})
    foreach(LIB ${ACML_LIBRARIES})
        set(ACML_LIBRARIES_FRIENDLY "${ACML_LIBRARIES_FRIENDLY}${LIB} ")
    endforeach(LIB)
    message(STATUS "ACML libraries: " ${ACML_LIBRARIES_FRIENDLY})
    unset(ACML_LIBRARIES_FRIENDLY)
endif(NOT ACML_FIND_QUIETLY)

#   Check that versions of ACML and compiler are compatible
#   ACML 5 requires GCC 4.6 or higher

if(CMAKE_Fortran_COMPILER_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES ${ACML_A_PATH})
    check_fortran_source_compiles("
      call xerbla(\"0\",0)
      end
    " ACML_COMPILES)
    if(NOT ACML_COMPILES)
        message(FATAL_ERROR "Compiler and ACML are not compatible")
    endif(NOT ACML_COMPILES)
endif(CMAKE_Fortran_COMPILER_WORKS)

#   Set USE_ACML definition

add_definitions(-DUSE_ACML)
set(USE_ACML TRUE)

else(ACML_FOUND)

if(ACML_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find ACML")
endif(ACML_FIND_REQUIRED)

endif(ACML_FOUND)

