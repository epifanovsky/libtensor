
set(SRC
	libtensor_performance_tests.C
	libtensor_pt_suite.C
	performance_test_suite.C
	test_expressions.C
)
add_executable(libtensor_performance_tests ${SRC})

add_definitions(-DLIBVMM_DEBUG)
add_definitions(-DLIBTENSOR_DEBUG)
add_definitions(-DLIBTENSOR_TIMINGS)
add_definitions(-g)

target_link_libraries(libtensor_performance_tests
	tensor vmm test
	${LIBS}
)

