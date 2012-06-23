set(SRC
    libtensor_performance_tests.C
    libtensor_pt_suite.C
    performance_test_suite.C
    test_expressions.C
)
add_executable(libtensor_performance_tests ${SRC})

include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

set_property(TARGET libtensor_performance_tests APPEND PROPERTY
    COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)
if(HAVE_LIBVMM)
    set_property(TARGET libtensor_performance_tests APPEND PROPERTY
        COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
else(HAVE_LIBVMM)
    set_property(TARGET libtensor_performance_tests APPEND PROPERTY
        COMPILE_DEFINITIONS WITHOUT_LIBVMM)
endif(HAVE_LIBVMM)

target_link_libraries(libtensor_performance_tests
    tensor ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} ${LIBUTIL_LIBS} test)

