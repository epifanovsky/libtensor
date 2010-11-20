set(SRC
	libtensor_tests.C
	libtensor_suite.C
	abs_index_test.C
	block_index_map_test.C
	block_index_space_test.C
	block_index_subspace_builder_test.C
	block_map_test.C
	block_tensor_test.C
	dimensions_test.C
	direct_block_tensor_test.C
	global_timings_test.C
	immutable_test.C
	index_test.C
	index_range_test.C
	mapped_block_tensor_test.C
	mask_test.C
	mp_safe_tensor_test.C
	orbit_test.C
	orbit_list_test.C
	permutation_test.C
	permutation_builder_test.C
	sequence_test.C
	symmetry_test.C
	symmetry_element_set_test.C
	task_batch_test.C
	tensor_test.C
	timer_test.C
	timings_test.C
	transf_list_test.C
	version_test.C
)

set(SRC_LINALG
	linalg/libtensor_linalg_tests.C
	linalg/libtensor_linalg_suite.C
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
	linalg/linalg_ij_ipq_jqp_x_test.C
	linalg/linalg_ijk_ipkq_jpq_x_test.C
	linalg/linalg_ijk_pikq_jpq_x_test.C
	linalg/linalg_ijk_piqk_jpq_x_test.C
	linalg/linalg_ijkl_ipl_kpj_x_test.C
	linalg/linalg_ijkl_ipkq_pljq_x_test.C
	linalg/linalg_ijkl_iplq_kpjq_x_test.C
	linalg/linalg_ijkl_iplq_pkjq_x_test.C
	linalg/linalg_ijkl_iplq_pkqj_x_test.C
	linalg/linalg_ijkl_ipqk_pljq_x_test.C
	linalg/linalg_ijkl_ipql_pkjq_x_test.C
	linalg/linalg_ijkl_ipql_pkqj_x_test.C
	linalg/linalg_ijkl_ipql_qkpj_x_test.C
	linalg/linalg_ijkl_pilq_kpjq_x_test.C
	linalg/linalg_ijkl_pilq_pkjq_x_test.C
	linalg/linalg_ijkl_pikq_pljq_x_test.C
	linalg/linalg_ijkl_piqk_pljq_x_test.C
	linalg/linalg_ijkl_piql_kpqj_x_test.C
	linalg/linalg_ijkl_piql_pkjq_x_test.C
	linalg/linalg_ijkl_piql_pkqj_x_test.C
	linalg/linalg_ijkl_piql_qkpj_x_test.C
	linalg/linalg_ijkl_pkiq_jplq_x_test.C
	linalg/linalg_ijkl_pkiq_jpql_x_test.C
	linalg/linalg_ijkl_pkiq_pjlq_x_test.C
	linalg/linalg_ijkl_pkiq_pjql_x_test.C
	linalg/linalg_ijkl_pliq_jpkq_x_test.C
	linalg/linalg_ijkl_pliq_jpqk_x_test.C
	linalg/linalg_ijkl_pliq_pjqk_x_test.C
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
	iface/expr_test.C
	iface/labeled_btensor_test.C
	iface/letter_expr_test.C
	iface/letter_test.C
	iface/mapped_btensor_test.C
	iface/symm_test.C
	iface/trace_test.C
)

file(GLOB SRC_BTOD btod/*.C)
file(GLOB SRC_SYMMETRY symmetry/*.C)
file(GLOB SRC_TOD tod/*.C)

set(TARGETS
	libtensor_tests
	libtensor_btod_tests
	libtensor_iface_tests
	libtensor_linalg_tests
	libtensor_symmetry_tests
	libtensor_tod_tests
)


add_executable(libtensor_tests ${SRC})
add_executable(libtensor_btod_tests ${SRC_BTOD})
add_executable(libtensor_iface_tests ${SRC_IFACE})
add_executable(libtensor_linalg_tests ${SRC_LINALG})
add_executable(libtensor_symmetry_tests ${SRC_SYMMETRY})
add_executable(libtensor_tod_tests ${SRC_TOD})

include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

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

