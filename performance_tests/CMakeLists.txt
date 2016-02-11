set(SRC
    libtensor_performance_tests.C
    libtensor_pt_suite.C
    performance_test_suite.C
    test_expressions.C
)
add_executable(libtensor_performance_tests ${SRC})
if(WITH_CTF)
    add_executable(libtensor_ctf_benchmarks libtensor_ctf_benchmarks.C)
endif(WITH_CTF)

include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

set_property(TARGET libtensor_performance_tests APPEND PROPERTY
    COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBTENSOR_DEBUG>)
set_property(TARGET libtensor_performance_tests APPEND PROPERTY
    LINK_FLAGS ${OpenMP_CXX_FLAGS})
if(HAVE_LIBVMM)
    set_property(TARGET libtensor_performance_tests APPEND PROPERTY
        COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBVMM_DEBUG>)
else(HAVE_LIBVMM)
    set_property(TARGET libtensor_performance_tests APPEND PROPERTY
        COMPILE_DEFINITIONS WITHOUT_LIBVMM)
endif(HAVE_LIBVMM)
if(WITH_CTF)
    set_property(TARGET libtensor_ctf_benchmarks APPEND PROPERTY
        COMPILE_DEFINITIONS WITHOUT_LIBVMM)
endif(WITH_CTF)

target_link_libraries(libtensor_performance_tests
    tensor tensor_core ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} ${LIBUTIL_LIBS}
    test)
if(WITH_CTF)
    set_property(TARGET libtensor_ctf_benchmarks APPEND PROPERTY
        LINK_FLAGS ${OpenMP_CXX_FLAGS})
    target_link_libraries(libtensor_ctf_benchmarks
        tensor tensor_core ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} ${LIBUTIL_LIBS})
endif(WITH_CTF)
