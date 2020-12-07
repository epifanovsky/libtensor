include(CheckCXXCompilerFlag)

macro(enable_if_cxx_compiles VARIABLE FLAG)
        # Checks whether a cxx compiler supports a flag and if yes
        # adds it to the variable provided.
        #
        string(REGEX REPLACE "[^a-zA-Z0-9]" "" FLAG_CLEAN "${FLAG}")
        CHECK_CXX_COMPILER_FLAG("-Werror ${FLAG}" HAVE_FLAG_${FLAG_CLEAN})
        if (DRB_HAVE_FLAG_${FLAG_CLEAN})
                set(${VARIABLE} "${${VARIABLE}} ${FLAG}")
        endif()
        unset(FLAG_CLEAN)
endmacro(enable_if_cxx_compiles)

# Standard flags for building libtensor, used without check
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-error -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-deprecated -Wno-unused-but-set-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds -Wno-maybe-uninitialized")

# Really useful and sensible flags to check for common errors
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Woverloaded-virtual")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wcast-align")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wuseless-cast")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wmisleading-indentation")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wduplicated-cond")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wduplicated-branches")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wlogical-op")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wnull-dereference")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wdouble-promotion")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wformat=2")

# These we rather want as warnings, not errors
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations")
