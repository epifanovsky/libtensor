cmake_minimum_required(VERSION 2.8.10)
project(LIBUTIL)

set(LIBUTIL_DIR ${LIBUTIL_SOURCE_DIR})
get_filename_component(LIBTEST_DIR ${LIBUTIL_DIR}/../libtest ABSOLUTE)

include(${LIBUTIL_DIR}/cmake/DepsLibUtil.txt)

add_subdirectory(libutil)
add_subdirectory(tests)

