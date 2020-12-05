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
    block_tensor/btod_contract3_test.C
    block_tensor/btod_copy_test.C
    block_tensor/btod_diag_test.C
    block_tensor/btod_diagonalize_test.C
    block_tensor/btod_dirsum_test.C
    block_tensor/btod_dotprod_test.C
    block_tensor/btod_ewmult2_test.C
    block_tensor/btod_extract_test.C
    block_tensor/btod_import_raw_test.C
    block_tensor/btod_import_raw_stream_test.C
    block_tensor/btod_mult_test.C
    block_tensor/btod_mult1_test.C
    block_tensor/btod_print_test.C
    block_tensor/btod_random_test.C
    block_tensor/btod_read_test.C
    block_tensor/btod_scale_test.C
    block_tensor/btod_select_test.C
    block_tensor/btod_set_diag_test.C
    block_tensor/btod_set_elem_test.C
    block_tensor/btod_set_test.C
    block_tensor/btod_shift_diag_test.C
    block_tensor/btod_sum_test.C
    block_tensor/btod_symcontract3_test.C
    block_tensor/btod_symmetrize2_test.C
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
    iface/expr_tensor_test.C
    iface/letter_expr_test.C
    iface/letter_test.C
    iface/mult_test.C
    iface/set_test.C
    iface/symm_test.C
    iface/trace_test.C
)

set(TARGETS
    # libtensor_block_tensor_tests
    libtensor_expr_tests
    libtensor_iface_tests
    libtensor_symmetry_tests
)


macro(libtensor_add_tests SUITE)
    foreach(TEST ${ARGN})
        add_executable(${TEST} ${TEST}.C)
        target_link_libraries(${TEST} tensorlight)
        target_include_directories(${TEST} PRIVATE ${libtensorlight_SOURCE_DIR})
        add_test(NAME "libtensor/${SUITE}/${TEST}" COMMAND ${TEST})
    endforeach()
endmacro()

add_subdirectory(linalg)
add_subdirectory(core)
add_subdirectory(symmetry)
add_subdirectory(dense_tensor)

