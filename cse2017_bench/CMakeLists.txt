set(SRC_BENCH
    dgemm_bench
)

foreach(BENCH ${SRC_BENCH})

    add_executable(${BENCH} ${BENCH}.C)

    set_property(TARGET ${BENCH} APPEND PROPERTY
        COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBTENSOR_DEBUG>)
    set_property(TARGET ${BENCH} APPEND PROPERTY
        LINK_FLAGS ${OpenMP_CXX_FLAGS})
    if(HAVE_LIBVMM)
        set_property(TARGET ${BENCH} APPEND PROPERTY
            COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBVMM_DEBUG>)
    else(HAVE_LIBVMM)
        set_property(TARGET ${BENCH} APPEND PROPERTY
            COMPILE_DEFINITIONS WITHOUT_LIBVMM)
    endif(HAVE_LIBVMM)

    target_link_libraries(${BENCH}
        tensor tensor_core ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} util)
    #if(WITH_CTF)
    #    set_property(TARGET libtensor_ctf_benchmarks APPEND PROPERTY
    #        LINK_FLAGS ${OpenMP_CXX_FLAGS})
    #    target_link_libraries(libtensor_ctf_benchmarks
    #        tensor tensor_core ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} util)
    #endif(WITH_CTF)

endforeach(BENCH)
