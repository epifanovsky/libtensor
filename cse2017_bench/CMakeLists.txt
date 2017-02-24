set(SRC_MKL_BENCH
    dgemm_bench
    dsymm_bench
)
set(SRC_BENCH
    contract_dgemm_bench
    contract_simple_bench
    contract_regular_bench
    contract_regular_symm_bench
    contract_regular_part_bench
    contract_vm_bench
    contract_vm_symm_bench
    contract_vm_part_bench
)
set(SRC_XM_BENCH
    contract_xm_bench
    contract_xm_symm_bench
    contract_xm_part_bench
)

foreach(BENCH ${SRC_MKL_BENCH})
    add_executable(${BENCH} ${BENCH}.C)
    set_property(TARGET ${BENCH} APPEND PROPERTY COMPILE_FLAGS "-mkl")
    set_property(TARGET ${BENCH} APPEND PROPERTY LINK_FLAGS "-mkl")
    target_link_libraries(${BENCH} util)
endforeach(BENCH)

foreach(BENCH ${SRC_XM_BENCH})
    add_executable(${BENCH} ${BENCH}.C)
    set_property(TARGET ${BENCH} APPEND PROPERTY COMPILE_FLAGS "-mkl")
    set_property(TARGET ${BENCH} APPEND PROPERTY LINK_FLAGS "-mkl")
    target_link_libraries(${BENCH}
        tensor tensor_core ${LIBTENSOR_DEP_LIBS} util)
endforeach(BENCH)

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
