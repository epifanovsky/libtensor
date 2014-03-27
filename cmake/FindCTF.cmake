#
#    Locates the Cyclops Tensor Framewok (CTF) library
#
#    Output:
#
#    CTF_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    CTF_PATH         - Library home.
#    CTF_INCLUDE_PATH - Path to the library's header files.
#    CTF_LIBRARY_PATH - Path to the library's binaries.
#    CTF_LIBRARIES    - Line for the linker.
#
#    The following locations are searched:
#    1. $ENV{CTF_DIR}
#    
set(CTF_FOUND FALSE)

set(CTF_DIR "$ENV{CTF_DIR}")

if(CTF_DIR)

find_file(CTF_HPP ctf.hpp PATHS "${CTF_DIR}/include" NO_DEFAULT_PATH)

set(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.lib")
find_library(CTF_A ctf PATHS "${CTF_DIR}/lib" NO_DEFAULT_PATH)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${SAVED_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(SAVED_CMAKE_FIND_LIBRARY_SUFFIXES)

if(NOT (CTF_HPP AND CTF_A))
    message(SEND_ERROR "CTF was not found in ${CTF_DIR}")
endif()

if(CTF_HPP AND CTF_A)

set(CTF_PATH "${CTF_DIR}")
set(CTF_INCLUDE_PATH "${CTF_DIR}/include")
set(CTF_LIBRARY_PATH "${CTF_DIR}/lib")
add_library(ctf STATIC IMPORTED)
set_target_properties(ctf PROPERTIES IMPORTED_LOCATION ${CTF_A})
set(CTF_LIBRARIES ctf)
set(CTF_FOUND TRUE)
if(NOT CTF_FIND_QUIETLY)
    message(STATUS "Found CTF: ${CTF_PATH}")
endif(NOT CTF_FIND_QUIETLY)

endif(CTF_HPP AND CTF_A)

endif(CTF_DIR)

