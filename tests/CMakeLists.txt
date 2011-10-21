set(SRC_CORE
	core/libtensor_core_tests.C
	core/libtensor_core_suite.C
	core/abs_index_test.C
	core/block_index_map_test.C
	core/block_index_space_test.C
	core/block_index_subspace_builder_test.C
	core/block_map_test.C
	core/block_tensor_test.C
	core/dimensions_test.C
	core/direct_block_tensor_test.C
	core/global_timings_test.C
	core/immutable_test.C
	core/index_test.C
	core/index_range_test.C
	core/mapped_block_tensor_test.C
	core/mask_test.C
	core/mp_safe_tensor_test.C
	core/orbit_test.C
	core/orbit_list_test.C
	core/permutation_test.C
	core/permutation_builder_test.C
	core/sequence_test.C
	core/symmetry_test.C
	core/symmetry_element_set_test.C
	core/task_batch_test.C
	core/tensor_test.C
	core/timer_test.C
	core/timings_test.C
	core/transf_list_test.C
	core/version_test.C
)

set(SRC_TOD
	tod/libtensor_tod_tests.C
	tod/libtensor_tod_suite.C
	tod/contraction2_list_builder_test.C
	tod/contraction2_test.C
	tod/tod_add_test.C
	tod/tod_apply_test.C
	tod/tod_btconv_test.C
	tod/tod_compare_test.C
	tod/tod_contract2_test.C
	tod/tod_copy_test.C
	tod/tod_diag_test.C
	tod/tod_dirsum_test.C
	tod/tod_dotprod_test.C
	tod/tod_ewmult2_test.C
	tod/tod_extract_test.C
	tod/tod_import_raw_test.C
	tod/tod_mult_test.C
	tod/tod_mult1_test.C
	tod/tod_random_test.C
	tod/tod_scale_test.C
	tod/tod_scatter_test.C
	tod/tod_select_test.C
	tod/tod_set_diag_test.C
	tod/tod_set_elem_test.C
	tod/tod_set_test.C
	tod/tod_sum_test.C
	tod/tod_symcontract2_test.C
	tod/tod_trace_test.C
)

set(SRC_SYMMETRY
	symmetry/libtensor_symmetry_tests.C
	symmetry/libtensor_symmetry_suite.C
	symmetry/partition_set_test.C
	symmetry/permutation_group_test.C
	symmetry/point_group_table_test.C
	symmetry/product_table_container_test.C
	symmetry/se_label_test.C
	symmetry/se_part_test.C
	symmetry/se_perm_test.C
	symmetry/so_add_impl_label_test.C
	symmetry/so_add_impl_part_test.C
	symmetry/so_add_impl_perm_test.C
	symmetry/so_add_test.C
	symmetry/so_apply_impl_label_test.C
	symmetry/so_apply_impl_part_test.C
	symmetry/so_apply_impl_perm_test.C
	symmetry/so_apply_test.C
	symmetry/so_concat_impl_label_test.C
	symmetry/so_concat_impl_part_test.C
	symmetry/so_concat_impl_perm_test.C
	symmetry/so_concat_test.C
	symmetry/so_copy_test.C
	symmetry/so_merge_impl_label_test.C
	symmetry/so_merge_impl_part_test.C
	symmetry/so_merge_impl_perm_test.C
	symmetry/so_merge_test.C
	symmetry/so_mult_impl_label_test.C
	symmetry/so_mult_impl_part_test.C
	symmetry/so_mult_impl_perm_test.C
	symmetry/so_mult_test.C
	symmetry/so_permute_impl_label_test.C
	symmetry/so_permute_impl_part_test.C
	symmetry/so_permute_impl_perm_test.C
	symmetry/so_proj_down_impl_label_test.C
	symmetry/so_proj_down_impl_part_test.C
	symmetry/so_proj_down_impl_perm_test.C
	symmetry/so_proj_down_test.C
	symmetry/so_proj_up_impl_label_test.C
	symmetry/so_proj_up_impl_part_test.C
	symmetry/so_proj_up_impl_perm_test.C
	symmetry/so_proj_up_test.C
	symmetry/so_stabilize_impl_label_test.C
	symmetry/so_stabilize_impl_part_test.C
	symmetry/so_stabilize_impl_perm_test.C
	symmetry/so_stabilize_test.C
	symmetry/so_symmetrize_test.C
	symmetry/symmetry_element_set_adapter_test.C
)

set(SRC_BTOD
	btod/libtensor_btod_tests.C
	btod/libtensor_btod_suite.C
	btod/addition_schedule_test.C
	btod/btod_add_test.C
	btod/btod_apply_test.C
	btod/btod_compare_test.C
	btod/btod_contract2_test.C
	btod/btod_copy_test.C
	btod/btod_diag_test.C
	btod/btod_diagonalize_test.C
	btod/btod_dirsum_test.C
	btod/btod_dotprod_test.C
	btod/btod_ewmult2_test.C
	btod/btod_extract_test.C
	btod/btod_import_raw_test.C
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
	btod/btod_symmetrize_test.C
	btod/btod_symmetrize3_test.C
	btod/btod_trace_test.C
	btod/btod_tridiagonalize_test.C
)

set(SRC_LINALG
	linalg/libtensor_linalg_tests.C
	linalg/libtensor_linalg_suite.C
    linalg/linalg_add_i_i_x_x_test.C
	linalg/linalg_x_p_p_test.C
	linalg/linalg_i_i_x_test.C
	linalg/linalg_x_pq_qp_test.C
	linalg/linalg_i_ip_p_x_test.C
	linalg/linalg_i_pi_p_x_test.C
	linalg/linalg_ij_i_j_x_test.C
	linalg/linalg_ij_ji_test.C
	linalg/linalg_i_ipq_qp_x_test.C
	linalg/linalg_ij_ip_jp_x_test.C
	linalg/linalg_ij_ip_pj_x_test.C
	linalg/linalg_ij_pi_jp_x_test.C
	linalg/linalg_ij_pi_pj_x_test.C
	linalg/linalg_ij_pji_p_x_test.C
	linalg/linalg_ij_ipq_jqp_x_test.C
	linalg/linalg_ijk_ip_pkj_x_test.C
	linalg/linalg_ijk_pi_pkj_x_test.C
	linalg/linalg_ijk_ipq_kjqp_x_test.C
	linalg/linalg_ijk_ipkq_jpq_x_test.C
	linalg/linalg_ijk_pikq_jpq_x_test.C
	linalg/linalg_ijk_piqk_jpq_x_test.C
    linalg/linalg_ijk_pkiq_pjq_x_test.C
    linalg/linalg_ijkl_ikp_jpl_x_test.C
    linalg/linalg_ijkl_ipj_plk_x_test.C
    linalg/linalg_ijkl_ipl_jpk_x_test.C
	linalg/linalg_ijkl_ipl_kpj_x_test.C
	linalg/linalg_ijkl_ipkq_pljq_x_test.C
	linalg/linalg_ijkl_iplq_kpjq_x_test.C
	linalg/linalg_ijkl_iplq_pkjq_x_test.C
	linalg/linalg_ijkl_iplq_pkqj_x_test.C
	linalg/linalg_ijkl_ipqk_pljq_x_test.C
	linalg/linalg_ijkl_ipql_pkjq_x_test.C
	linalg/linalg_ijkl_ipql_pkqj_x_test.C
	linalg/linalg_ijkl_ipql_qkpj_x_test.C
	linalg/linalg_ijkl_ipql_qpkj_x_test.C
	linalg/linalg_ijkl_pilq_kpjq_x_test.C
	linalg/linalg_ijkl_pilq_pkjq_x_test.C
	linalg/linalg_ijkl_pikq_pljq_x_test.C
	linalg/linalg_ijkl_piqk_pljq_x_test.C
	linalg/linalg_ijkl_piql_kpqj_x_test.C
	linalg/linalg_ijkl_piql_pkjq_x_test.C
	linalg/linalg_ijkl_piql_pkqj_x_test.C
	linalg/linalg_ijkl_piql_qkpj_x_test.C
	linalg/linalg_ijkl_piql_qpkj_x_test.C
	linalg/linalg_ijkl_pkiq_jplq_x_test.C
	linalg/linalg_ijkl_pkiq_jpql_x_test.C
	linalg/linalg_ijkl_pkiq_pjlq_x_test.C
	linalg/linalg_ijkl_pkiq_pjql_x_test.C
	linalg/linalg_ijkl_pliq_jpkq_x_test.C
	linalg/linalg_ijkl_pliq_jpqk_x_test.C
	linalg/linalg_ijkl_pliq_pjqk_x_test.C
	linalg/linalg_ijklm_ipkm_jlp_x_test.C
)

set(SRC_IFACE
	iface/libtensor_iface_tests.C
	iface/libtensor_iface_suite.C
	iface/anon_eval_test.C
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
	iface/ewmult_test.C
	iface/expr_test.C
	iface/labeled_btensor_test.C
	iface/letter_expr_test.C
	iface/letter_test.C
	iface/mapped_btensor_test.C
	iface/mult_test.C
	iface/symm_test.C
	iface/trace_test.C
)

set(SRC_CUDA
    tod/libtensor_tod_cuda_tests.C
    tod/libtensor_tod_cuda_suite.C
    tod/tod_cuda_copy_test.cu 
)

set(TARGETS
	libtensor_core_tests
	libtensor_tod_tests
	libtensor_btod_tests
	libtensor_iface_tests
	libtensor_linalg_tests
	libtensor_symmetry_tests
)


add_executable(libtensor_core_tests ${SRC_CORE})
add_executable(libtensor_btod_tests ${SRC_BTOD})
add_executable(libtensor_iface_tests ${SRC_IFACE})
add_executable(libtensor_linalg_tests ${SRC_LINALG})
add_executable(libtensor_symmetry_tests ${SRC_SYMMETRY})
add_executable(libtensor_tod_tests ${SRC_TOD})

include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

if(CUDA_FOUND)
    set(TARGETS ${TARGETS} libtensor_cuda_tests)
    cuda_add_executable(libtensor_cuda_tests ${SRC_CUDA})
endif(CUDA_FOUND)

set_property(TARGET ${TARGETS} APPEND PROPERTY
	COMPILE_DEFINITIONS LIBTENSOR_INSTANTIATE_TEMPLATES)
set_property(TARGET ${TARGETS} APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
set_property(TARGET ${TARGETS} APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)

foreach(TARGET ${TARGETS})
target_link_libraries(${TARGET}
	tensor ${LIBTENSOR_DEP_LIBS} ${LIBVMM_LIBS} test)
endforeach(TARGET)

