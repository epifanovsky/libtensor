set(SRC
	libtensor_performance_tests.C
	libtensor_pt_suite.C
	performance_test_suite.C
	test_expressions.C
)
add_executable(libtensor_performance_tests ${SRC})

include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

set_target_properties(libtensor_performance_tests PROPERTIES
	COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG
	COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)

target_link_libraries(libtensor_performance_tests
	tensor ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} test)

