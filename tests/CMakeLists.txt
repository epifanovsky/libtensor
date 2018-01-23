set(SRC_DIAG_TENSOR
    diag_tensor/libtensor_diag_tensor_tests.C
    diag_tensor/libtensor_diag_tensor_suite.C
    diag_tensor/diag_tensor_test.C
    diag_tensor/diag_tensor_space_test.C
    diag_tensor/diag_tensor_subspace_test.C
    diag_tensor/diag_to_add_space_test.C
    diag_tensor/diag_to_contract2_space_test.C
    diag_tensor/diag_tod_adjust_space_test.C
    diag_tensor/diag_tod_contract2_test.C
    diag_tensor/diag_tod_contract2_part_test.C
    diag_tensor/diag_tod_copy_test.C
    diag_tensor/diag_tod_dotprod_test.C
    diag_tensor/diag_tod_mult1_test.C
    diag_tensor/diag_tod_random_test.C
    diag_tensor/diag_tod_set_test.C
    diag_tensor/tod_conv_diag_tensor_test.C
)

set(SRC_DIAG_BLOCK_TENSOR
    diag_block_tensor/libtensor_diag_block_tensor_tests.C
    diag_block_tensor/libtensor_diag_block_tensor_suite.C
    diag_block_tensor/diag_block_tensor_test.C
    diag_block_tensor/diag_btod_contract2_test.C
    diag_block_tensor/diag_btod_copy_test.C
    diag_block_tensor/diag_btod_random_test.C
    diag_block_tensor/tod_conv_diag_block_tensor_test.C
)

set(SRC_CTF_DENSE_TENSOR
    ctf_dense_tensor/libtensor_ctf_dense_tensor_tests.C
    ctf_dense_tensor/libtensor_ctf_dense_tensor_suite.C
    ctf_dense_tensor/ctf_dense_tensor_test.C
    ctf_dense_tensor/ctf_symmetry_test.C
    ctf_dense_tensor/ctf_tod_contract2_test.C
    ctf_dense_tensor/ctf_tod_copy_test.C
    ctf_dense_tensor/ctf_tod_diag_test.C
    ctf_dense_tensor/ctf_tod_dirsum_test.C
    ctf_dense_tensor/ctf_tod_distribute_test.C
    ctf_dense_tensor/ctf_tod_dotprod_test.C
    ctf_dense_tensor/ctf_tod_ewmult2_test.C
    ctf_dense_tensor/ctf_tod_mult_test.C
    ctf_dense_tensor/ctf_tod_mult1_test.C
    ctf_dense_tensor/ctf_tod_random_test.C
    ctf_dense_tensor/ctf_tod_scale_test.C
    ctf_dense_tensor/ctf_tod_scatter_test.C
    ctf_dense_tensor/ctf_tod_set_test.C
    ctf_dense_tensor/ctf_tod_set_diag_test.C
    ctf_dense_tensor/ctf_tod_set_symmetry_test.C
    ctf_dense_tensor/ctf_tod_trace_test.C
)

set(SRC_CTF_BLOCK_TENSOR
    ctf_block_tensor/libtensor_ctf_block_tensor_tests.C
    ctf_block_tensor/libtensor_ctf_block_tensor_suite.C
    ctf_block_tensor/ctf_btod_collect_test.C
    ctf_block_tensor/ctf_btod_contract2_test.C
    ctf_block_tensor/ctf_btod_copy_test.C
    ctf_block_tensor/ctf_btod_diag_test.C
    ctf_block_tensor/ctf_btod_dirsum_test.C
    ctf_block_tensor/ctf_btod_distribute_test.C
    ctf_block_tensor/ctf_btod_dotprod_test.C
    ctf_block_tensor/ctf_btod_ewmult2_test.C
    ctf_block_tensor/ctf_btod_mult_test.C
    ctf_block_tensor/ctf_btod_mult1_test.C
    ctf_block_tensor/ctf_btod_random_test.C
    ctf_block_tensor/ctf_btod_scale_test.C
    ctf_block_tensor/ctf_btod_set_test.C
    ctf_block_tensor/ctf_btod_symmetrize2_test.C
    ctf_block_tensor/ctf_btod_symmetrize3_test.C
    ctf_block_tensor/ctf_btod_trace_test.C
    ctf_block_tensor/ctf_symmetry_builder_test.C
)

set(SRC_SYMMETRY
    symmetry/adjacency_list_test.C
    symmetry/block_labeling_test.C
    symmetry/combine_label_test.C
    symmetry/combine_part_test.C
    symmetry/eval_sequence_list_test.C
    symmetry/er_merge_test.C
    symmetry/er_optimize_test.C
    symmetry/er_reduce_test.C
    symmetry/evaluation_rule_test.C
    symmetry/libtensor_symmetry_tests.C
    symmetry/libtensor_symmetry_suite.C
    symmetry/permutation_group_test.C
    symmetry/point_group_table_test.C
    symmetry/product_rule_test.C
    symmetry/product_table_container_test.C
    symmetry/se_label_test_base.C
    symmetry/se_label_test.C
    symmetry/se_part_test.C
    symmetry/se_perm_test.C
    symmetry/so_apply_se_label_test.C
    symmetry/so_apply_se_part_test.C
    symmetry/so_apply_se_perm_test.C
    symmetry/so_apply_test.C
    symmetry/so_copy_test.C
    symmetry/so_dirprod_se_label_test.C
    symmetry/so_dirprod_se_part_test.C
    symmetry/so_dirprod_se_perm_test.C
    symmetry/so_dirprod_test.C
    symmetry/so_dirsum_se_label_test.C
    symmetry/so_dirsum_se_part_test.C
    symmetry/so_dirsum_se_perm_test.C
    symmetry/so_dirsum_test.C
    symmetry/so_merge_se_label_test.C
    symmetry/so_merge_se_part_test.C
    symmetry/so_merge_se_perm_test.C
    symmetry/so_merge_test.C
    symmetry/so_permute_se_label_test.C
    symmetry/so_permute_se_part_test.C
    symmetry/so_permute_se_perm_test.C
    symmetry/so_reduce_se_label_test.C
    symmetry/so_reduce_se_part_test.C
    symmetry/so_reduce_se_perm_test.C
    symmetry/so_reduce_test.C
    symmetry/so_symmetrize_se_label_test.C
    symmetry/so_symmetrize_se_part_test.C
    symmetry/so_symmetrize_test.C
    symmetry/symmetry_element_set_adapter_test.C
)

set(SRC_BLOCK_TENSOR
    block_tensor/libtensor_block_tensor_tests.C
    block_tensor/libtensor_block_tensor_suite.C
    block_tensor/block_tensor_test.C
    block_tensor/direct_block_tensor_test.C
    block_tensor/addition_schedule_test.C
    block_tensor/bto_contract2_bis_test.C
    block_tensor/bto_contract2_sym_test.C
    block_tensor/btod_add_test.C
    block_tensor/btod_apply_test.C
    block_tensor/btod_compare_test.C
    block_tensor/btod_contract2_test.C
    block_tensor/bto_contract2_test.C
    block_tensor/btod_contract3_test.C
    block_tensor/btod_copy_test.C
    block_tensor/bto_copy_test.C
    block_tensor/btod_diag_test.C
    block_tensor/bto_diag_test.C
    block_tensor/btod_diagonalize_test.C
    block_tensor/btod_dirsum_test.C
    block_tensor/bto_dirsum_test.C
    block_tensor/btod_dotprod_test.C
    block_tensor/bto_dotprod_test.C
    block_tensor/btod_ewmult2_test.C
    block_tensor/bto_ewmult2_test.C
    block_tensor/btod_extract_test.C
    block_tensor/btod_import_raw_test.C
    block_tensor/btod_import_raw_stream_test.C
    block_tensor/btod_mult_test.C
    block_tensor/btod_mult1_test.C
    block_tensor/btod_print_test.C
    block_tensor/btod_random_test.C
    block_tensor/bto_random_test.C
    block_tensor/btod_read_test.C
    block_tensor/btod_scale_test.C
    block_tensor/bto_scale_test.C
    block_tensor/btod_select_test.C
    block_tensor/btod_set_diag_test.C
    block_tensor/btod_set_elem_test.C
    block_tensor/btod_set_test.C
    block_tensor/bto_set_test.C
    block_tensor/btod_shift_diag_test.C
    block_tensor/btod_sum_test.C
    block_tensor/btod_symcontract3_test.C
    block_tensor/btod_symmetrize2_test.C
    block_tensor/bto_symmetrize2_test.C
    block_tensor/btod_symmetrize3_test.C
    block_tensor/btod_symmetrize4_test.C
    block_tensor/btod_trace_test.C
    block_tensor/btod_tridiagonalize_test.C
    block_tensor/btod_vmpriority_test.C
    block_tensor/gen_bto_aux_add_test.C
    block_tensor/gen_bto_aux_copy_test.C
    block_tensor/gen_bto_contract2_clst_builder_test.C
    block_tensor/gen_bto_dirsum_sym_test.C
    block_tensor/gen_bto_symcontract2_sym_test.C
    block_tensor/gen_bto_unfold_symmetry_test.C
)

if(USE_LIBXM)
    add_definitions(-DUSE_LIBXM)
    set(SRC_BLOCK_TENSOR ${SRC_BLOCK_TENSOR}
        block_tensor/btod_contract2_xm_test.C)
endif(USE_LIBXM)

set(SRC_EXPR
    expr/libtensor_expr_tests.C
    expr/libtensor_expr_suite.C
    expr/expr_tree_test.C
    expr/graph_test.C
    expr/node_add_test.C
    expr/node_contract_test.C
    expr/node_diag_test.C
    expr/node_dot_product_test.C
    expr/node_ident_any_tensor_test.C
    expr/node_product_test.C
    expr/node_scalar_test.C
    expr/node_set_test.C
    expr/node_trace_test.C
    expr/node_transform_test.C
)

set(SRC_IFACE
    iface/libtensor_iface_tests.C
    iface/libtensor_iface_suite.C
    iface/any_tensor_test.C
    iface/bispace_expr_test.C
    iface/bispace_test.C
    iface/btensor_test.C
    iface/contract_test.C
    iface/diag_test.C
    iface/direct_product_test.C
    iface/dirsum_test.C
    iface/dot_product_test.C
    iface/eval_btensor_double_test.C
    iface/eval_register_test.C
    iface/ewmult_test.C
    iface/expr_test.C
    iface/expr_test_float_double.C
    iface/expr_tensor_test.C
    iface/letter_expr_test.C
    iface/letter_test.C
    iface/mult_test.C
    iface/set_test.C
    iface/symm_test.C
    iface/trace_test.C
)

set(SRC_CTF_IFACE
    ctf_iface/libtensor_ctf_iface_tests.C
    ctf_iface/libtensor_ctf_iface_suite.C
    ctf_iface/ctf_btensor_test.C
    ctf_iface/ctf_dot_product_test.C
    ctf_iface/ctf_expr_test.C
    ctf_iface/ctf_mult_test.C
    ctf_iface/ctf_set_test.C
    ctf_iface/ctf_trace_test.C
)

set(SRC_BLOCK_SPARSE
    block_sparse/libtensor_block_sparse_tests.C
    block_sparse/libtensor_block_sparse_suite.C
    block_sparse/sparse_bispace_test.C
    block_sparse/sparse_btensor_test.C
    block_sparse/block_kernels_test.C
    block_sparse/block_loop_test.C
    block_sparse/sparse_loop_list_test.C
    block_sparse/sparsity_fuser_test.C
    block_sparse/sparse_loop_grouper_test.C
    block_sparse/sparse_block_tree_iterator_test.C
    block_sparse/sparse_block_tree_test.C
    block_sparse/direct_sparse_btensor_test.C
    block_sparse/blas_isomorphism_test.C
    block_sparse/batch_kernels_test.C
    block_sparse/batch_list_builder_test.C
    block_sparse/subspace_iterator_test.C
    block_sparse/batch_provider_test.C

    block_sparse/test_fixtures/permute_3d_sparse_120_test_f.C
    block_sparse/test_fixtures/contract2_test_f.C
)

set(SRC_BLOCK_SPARSE_BENCHMARK
    block_sparse/libtensor_block_sparse_benchmark.C 
    block_sparse/timer.C
)

set(TARGETS
    libtensor_diag_tensor_tests
    libtensor_diag_block_tensor_tests
    libtensor_block_tensor_tests
    libtensor_expr_tests
    libtensor_iface_tests
    libtensor_symmetry_tests
#    libtensor_block_sparse_benchmark
#    libtensor_block_sparse_tests
#    sparse_block_tree_benchmark
)
if(WITH_CTF)
    set(TARGETS ${TARGETS}
        libtensor_ctf_dense_tensor_tests
        libtensor_ctf_block_tensor_tests
        libtensor_ctf_iface_tests
    )
endif(WITH_CTF)

#add_executable(libtensor_block_tensor_tests ${SRC_BLOCK_TENSOR})
#add_executable(libtensor_expr_tests ${SRC_EXPR})
#add_executable(libtensor_iface_tests ${SRC_IFACE})
#add_executable(libtensor_symmetry_tests ${SRC_SYMMETRY})
#add_executable(libtensor_diag_tensor_tests ${SRC_DIAG_TENSOR})
#add_executable(libtensor_diag_block_tensor_tests ${SRC_DIAG_BLOCK_TENSOR})

#add_executable(libtensor_block_sparse_benchmark ${SRC_BLOCK_SPARSE_BENCHMARK})
#add_executable(libtensor_block_sparse_tests ${SRC_BLOCK_SPARSE})
#add_executable(sparse_block_tree_benchmark block_sparse/sparse_block_tree_benchmark.C block_sparse/timer.C)
if(WITH_CTF)
    set_property(SOURCE ctf_dense_tensor/ctf_dense_tensor_test.C
        ctf_dense_tensor/ctf_symmetry_test.C
        APPEND PROPERTY COMPILE_FLAGS "-std=c++0x")
    add_executable(libtensor_ctf_dense_tensor_tests ${SRC_CTF_DENSE_TENSOR})
#    set_property(TARGET libtensor_ctf_dense_tensor_tests APPEND PROPERTY
#        LINK_FLAGS ${OpenMP_CXX_FLAGS})
    add_executable(libtensor_ctf_block_tensor_tests ${SRC_CTF_BLOCK_TENSOR})
#    set_property(TARGET libtensor_ctf_block_tensor_tests APPEND PROPERTY
#        LINK_FLAGS ${OpenMP_CXX_FLAGS})
    add_executable(libtensor_ctf_iface_tests ${SRC_CTF_IFACE})
#    set_property(TARGET libtensor_ctf_iface_tests APPEND PROPERTY
#        LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif(WITH_CTF)

if(LIBTENSOR_PACKAGE)
    include_directories("${LIBTEST_DIR}")
else(LIBTENSOR_PACKAGE)
#    include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)
endif(LIBTENSOR_PACKAGE)

#set_property(TARGET ${TARGETS} APPEND PROPERTY
#    COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBTENSOR_DEBUG>)
#if(HAVE_LIBVMM)
#    set_property(TARGET ${TARGETS} APPEND PROPERTY
#        COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBTENSOR_DEBUG>)
#else(HAVE_LIBVMM)
#    set_property(TARGET ${TARGETS} APPEND PROPERTY
#        COMPILE_DEFINITIONS WITHOUT_LIBVMM)
#endif(HAVE_LIBVMM)

#foreach(TARGET ${TARGETS})
#target_link_libraries(${TARGET}
#    tensor tensor_core tensor_block_sparse
#    ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} libtest)
#set_property(TARGET ${TARGET} APPEND PROPERTY LINK_FLAGS ${OpenMP_CXX_FLAGS})
#endforeach(TARGET)

# Need custom flags for this so that expensive bounds checking code is not enabled, so we add it after
# Want optimization but also debug symbols for profiling
#set_property(TARGET libtensor_block_sparse_benchmark APPEND PROPERTY CMAKE_CXX_FLAGS_RELEASE -g)
#set_property(TARGET sparse_block_tree_benchmark  APPEND PROPERTY CMAKE_CXX_FLAGS_RELEASE -g)


macro(libtensor_add_tests SUITE)
    foreach(TEST ${ARGN})
        add_executable(${TEST} ${TEST}.C)
        target_link_libraries(${TEST} tensor)
        add_test(NAME "libtensor/${SUITE}/${TEST}" COMMAND ${TEST})
    endforeach()
endmacro()

add_subdirectory(linalg)
add_subdirectory(core)
add_subdirectory(symmetry)
add_subdirectory(dense_tensor)

