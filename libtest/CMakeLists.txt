cmake_minimum_required(VERSION 2.6)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project(LIBTEST)

set(LIBTEST_DIR ${LIBTEST_SOURCE_DIR})

add_subdirectory(libtest)
add_subdirectory(tests)

