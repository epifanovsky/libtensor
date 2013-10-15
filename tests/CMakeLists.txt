set(SRC_CORE
    core/libtensor_core_tests.C
    core/libtensor_core_suite.C
    core/abs_index_test.C
    core/block_index_space_test.C
    core/block_index_space_product_builder_test.C
    core/block_index_subspace_builder_test.C
    core/block_map_test.C
    core/block_tensor_test.C
    core/combined_orbits_test.C
    core/contraction2_list_builder_test.C
    core/contraction2_test.C
    core/dense_tensor_test.C
    core/dimensions_test.C
    core/direct_block_tensor_test.C
    core/immutable_test.C
    core/index_test.C
    core/index_range_test.C
    core/magic_dimensions_test.C
    core/mask_test.C
    core/orbit_test.C
    core/orbit_list_test.C
    core/permutation_test.C
    core/permutation_builder_test.C
    core/permutation_generator_test.C
    core/sequence_test.C
    core/sequence_generator_test.C
    core/short_orbit_test.C
    core/subgroup_orbits_test.C
    core/symmetry_test.C
    core/symmetry_element_set_test.C
    core/transf_list_test.C
    core/version_test.C
)

set(SRC_TOD
    tod/libtensor_tod_tests.C
    tod/libtensor_tod_suite.C
    tod/tod_add_test.C
    tod/tod_apply_test.C
    tod/tod_btconv_test.C
    tod/tod_compare_test.C
    tod/to_contract2_dims_test.C
    tod/tod_contract2_test.C
    tod/tod_copy_test.C
    tod/tod_copy_wnd_test.C
    tod/tod_diag_test.C
    tod/tod_dirsum_test.C
    tod/tod_dotprod_test.C
    tod/tod_ewmult2_test.C
    tod/tod_extract_test.C
    tod/tod_import_raw_test.C
    tod/tod_import_raw_stream_test.C
    tod/tod_mult_test.C
    tod/tod_mult1_test.C
    tod/tod_random_test.C
    tod/tod_scale_test.C
    tod/tod_scatter_test.C
    tod/tod_select_test.C
    tod/tod_set_diag_test.C
    tod/tod_set_elem_test.C
    tod/tod_set_test.C
    tod/tod_size_test.C
    tod/tod_trace_test.C
    tod/tod_vmpriority_test.C
)

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

set(SRC_BTOD
    btod/libtensor_btod_tests.C
    btod/libtensor_btod_suite.C
    btod/addition_schedule_test.C
    btod/bto_contract2_bis_test.C
    btod/bto_contract2_sym_test.C
    btod/btod_add_test.C
    btod/btod_apply_test.C
    btod/btod_compare_test.C
    btod/btod_contract2_test.C
    btod/btod_contract3_test.C
    btod/btod_copy_test.C
    btod/btod_diag_test.C
    btod/btod_diagonalize_test.C
    btod/btod_dirsum_test.C
    btod/btod_dotprod_test.C
    btod/btod_ewmult2_test.C
    btod/btod_extract_test.C
    btod/btod_import_raw_test.C
    btod/btod_import_raw_stream_test.C
    btod/btod_mult_test.C
    btod/btod_mult1_test.C
    btod/btod_print_test.C
    btod/btod_random_test.C
    btod/btod_read_test.C
    btod/btod_scale_test.C
    btod/btod_select_test.C
    btod/btod_set_diag_test.C
    btod/btod_set_elem_test.C
    btod/btod_set_test.C
    btod/btod_sum_test.C
    btod/btod_symcontract3_test.C
    btod/btod_symmetrize2_test.C
    btod/btod_symmetrize3_test.C
    btod/btod_symmetrize4_test.C
    btod/btod_trace_test.C
    btod/btod_tridiagonalize_test.C
    btod/btod_vmpriority_test.C
    btod/gen_bto_aux_add_test.C
    btod/gen_bto_aux_copy_test.C
    btod/gen_bto_contract2_clst_builder_test.C
    btod/gen_bto_dirsum_sym_test.C
    btod/gen_bto_symcontract2_sym_test.C
    btod/gen_bto_unfold_symmetry_test.C
)

set(SRC_LINALG
    linalg/libtensor_linalg_tests.C
    linalg/libtensor_linalg_suite.C
    linalg/linalg_add_i_i_x_x_test.C
    linalg/linalg_copy_ij_ji_test.C
    linalg/linalg_mul2_x_p_p_test.C
    linalg/linalg_mul2_i_i_i_x_test.C
    linalg/linalg_mul2_i_i_x_test.C
    linalg/linalg_mul2_x_pq_qp_test.C
    linalg/linalg_mul2_i_ip_p_x_test.C
    linalg/linalg_mul2_i_pi_p_x_test.C
    linalg/linalg_mul2_ij_i_j_x_test.C
    linalg/linalg_mul2_i_ipq_qp_x_test.C
    linalg/linalg_mul2_ij_ip_jp_x_test.C
    linalg/linalg_mul2_ij_ip_pj_x_test.C
    linalg/linalg_mul2_ij_pi_jp_x_test.C
    linalg/linalg_mul2_ij_pi_pj_x_test.C
)

set(SRC_EXPR
    expr/libtensor_expr_tests.C
    expr/libtensor_expr_suite.C
    expr/eval_plan_test.C
    expr/node_add_test.C
    expr/node_contract_test.C
    expr/node_diag_test.C
    expr/node_dirprod_test.C
    expr/node_dirsum_test.C
    expr/node_ewmult_test.C
    expr/node_mult_test.C
    expr/node_transform_test.C
)

set(SRC_IFACE
    iface/libtensor_iface_tests.C
    iface/libtensor_iface_suite.C
    iface/anon_eval_test.C
    iface/any_tensor_test.C
    iface/bispace_expr_test.C
    iface/bispace_test.C
    iface/btensor_test.C
    iface/contract_test.C
    iface/diag_test.C
    iface/direct_btensor_test.C
    iface/direct_eval_test.C
    iface/direct_product_test.C
    iface/dirsum_test.C
    iface/dot_product_test.C
    iface/eval_btensor_double_test.C
    iface/ewmult_test.C
    iface/expr_test.C
    iface/expr_tensor_test.C
#    iface/labeled_btensor_test.C
    iface/letter_expr_test.C
    iface/letter_test.C
    iface/mult_test.C
    iface/symm_test.C
    iface/tensor_list_test.C
    iface/trace_test.C
)

set(TARGETS
    libtensor_core_tests
    libtensor_tod_tests
    libtensor_diag_tensor_tests
    libtensor_diag_block_tensor_tests
    libtensor_btod_tests
    libtensor_expr_tests
    libtensor_iface_tests
    libtensor_linalg_tests
    libtensor_symmetry_tests
)

add_executable(libtensor_core_tests ${SRC_CORE})
add_executable(libtensor_btod_tests ${SRC_BTOD})
add_executable(libtensor_expr_tests ${SRC_EXPR})
add_executable(libtensor_iface_tests ${SRC_IFACE})
add_executable(libtensor_linalg_tests ${SRC_LINALG})
add_executable(libtensor_symmetry_tests ${SRC_SYMMETRY})
add_executable(libtensor_tod_tests ${SRC_TOD})
add_executable(libtensor_diag_tensor_tests ${SRC_DIAG_TENSOR})
add_executable(libtensor_diag_block_tensor_tests ${SRC_DIAG_BLOCK_TENSOR})

include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

set_property(TARGET ${TARGETS} APPEND PROPERTY
    COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)
if(HAVE_LIBVMM)
    set_property(TARGET ${TARGETS} APPEND PROPERTY
        COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
else(HAVE_LIBVMM)
    set_property(TARGET ${TARGETS} APPEND PROPERTY
        COMPILE_DEFINITIONS WITHOUT_LIBVMM)
endif(HAVE_LIBVMM)

foreach(TARGET ${TARGETS})
target_link_libraries(${TARGET}
    tensor ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} ${LIBUTIL_LIBS} test)
endforeach(TARGET)

