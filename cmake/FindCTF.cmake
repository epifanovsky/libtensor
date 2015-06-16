#
#    Locates the Cyclops Tensor Framewok (CTF) library
#
#    Input:
#
#    CTF_INCLUDE      - CTF include path
#    CTF_LIB          - CTF lib path
#
#    Output:
#
#    CTF_FOUND - TRUE/FALSE - Whether the library has been found.
#        If FALSE, all other output variables are not defined.
#
#    CTF_INCLUDE_PATH - Path to the library's header files.
#    CTF_LIBRARY_PATH - Path to the library's binaries.
#    CTF_LIBRARIES    - Line for the linker.
#
set(CTF_FOUND FALSE)

if(CTF_INCLUDE AND CTF_LIB)

    find_file(CTF_HPP ctf.hpp PATHS "${CTF_INCLUDE}" NO_DEFAULT_PATH)
    find_library(CTF_A ctf PATHS "${CTF_LIB}" NO_DEFAULT_PATH)
    if(NOT (CTF_HPP AND CTF_A))
        message(SEND_ERROR "CTF was not found in ${CTF_INCLUDE}")
    endif()

    if(CTF_HPP AND CTF_A)

        set(CTF_INCLUDE_PATH "${CTF_INCLUDE}")
        set(CTF_LIBRARY_PATH "${CTF_LIB}")
        add_library(ctf UNKNOWN IMPORTED)
        set_target_properties(ctf PROPERTIES IMPORTED_LOCATION ${CTF_A})
        set(CTF_LIBRARIES ctf)
        set(CTF_FOUND TRUE)
        if(NOT CTF_FIND_QUIETLY)
            message(STATUS "Found CTF: ${CTF_A}")
        endif(NOT CTF_FIND_QUIETLY)

    endif(CTF_HPP AND CTF_A)

endif()

