add_library(libtest test_exception.C test_suite.C unit_test.C)
set_target_properties(libtest PROPERTIES OUTPUT_NAME test)

