set(SRC
    libtest_suite.C
    libtest_tests.C
    test_suite_test.C
)

include_directories(${LIBTEST_DIR})
add_executable(libtest_tests ${SRC})
target_link_libraries(libtest_tests libtest)

