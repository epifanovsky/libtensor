set(SRC
	backtrace.C
	blas.C
	defs.C
	exception.C
	global_timings.C
	timer.C
	version.C
	btod/btod_diagonalize.C
	btod/btod_tridiagonalize.C
	symmetry/point_group_table.C
	symmetry/product_table_container.C
	tod/loop_list_add.C
	tod/loop_list_copy.C
	tod/loop_list_mul.C
	tod/loop_list_elem.C
	tod/loop_list_elem1.C
	tod/kernels/kern_mul_generic.C
	tod/kernels/kern_mul_i_i_x.C
	tod/kernels/kern_mul_i_ip_p.C
	tod/kernels/kern_mul_i_ipq_qp.C
	tod/kernels/kern_mul_i_p_ip.C
	tod/kernels/kern_mul_i_p_pi.C
	tod/kernels/kern_mul_i_pi_p.C
	tod/kernels/kern_mul_i_x_i.C
	tod/kernels/kern_mul_ij_ip_jp.C
	tod/kernels/kern_mul_ij_ip_pj.C
	tod/kernels/kern_mul_ij_jp_ip.C
	tod/kernels/kern_mul_ij_jp_pi.C
	tod/kernels/kern_mul_ij_jpq_iqp.C
	tod/kernels/kern_mul_ij_pi_jp.C
	tod/kernels/kern_mul_ij_pi_pj.C
	tod/kernels/kern_mul_ij_piq_pjq.C
	tod/kernels/kern_mul_ij_pj_ip.C
	tod/kernels/kern_mul_ij_pj_pi.C
	tod/kernels/kern_mul_ij_pjq_piq.C
	tod/kernels/kern_mul_ijk_ip_jpk.C
	tod/kernels/kern_mul_ijk_ipk_pj.C
	tod/kernels/kern_mul_ijk_jp_ipk.C
	tod/kernels/kern_mul_ijk_jpq_piqk.C
	tod/kernels/kern_mul_ijk_kp_ipj.C
	tod/kernels/kern_mul_ijk_kp_jpi.C
	tod/kernels/kern_mul_ijk_piq_jpkq.C
	tod/kernels/kern_mul_ijk_piq_jpqk.C
	tod/kernels/kern_mul_ijk_piq_pjkq.C
	tod/kernels/kern_mul_ijk_piq_pjqk.C
	tod/kernels/kern_mul_ijk_pjq_ipqk.C
	tod/kernels/kern_mul_ijk_pjq_piqk.C
	tod/kernels/kern_mul_ijk_pkiq_pjq.C
	tod/kernels/kern_mul_ijk_pkq_ipqj.C
	tod/kernels/kern_mul_ijk_pkq_piqj.C
	tod/kernels/kern_mul_ijkl_pkiq_jplq.C
	tod/kernels/kern_mul_ijkl_pkiq_pjlq.C
	tod/kernels/kern_mul_ijkl_pkjq_ipql.C
	tod/kernels/kern_mul_ijkl_pliq_jpkq.C
	tod/kernels/kern_mul_ijkl_pljq_ipkq.C
	tod/kernels/kern_mul_ijkl_pljq_pikq.C
	tod/kernels/kern_mul_x_p_p.C
	tod/kernels/kern_mul_x_pq_qp.C
)

set(SRC_LINALG
	linalg/generic/linalg_base_level1_generic.C
	linalg/generic/linalg_base_level2_generic.C
	linalg/generic/linalg_base_level3_generic.C
	linalg/generic/linalg_base_level4_generic.C
	linalg/generic/linalg_base_level5_generic.C
	linalg/generic/linalg_base_level6_generic.C
)

set(SRC_INST
	tod/tod_add.C
	tod/tod_contract2.C
	btod/additive_btod.C
	btod/basic_btod.C
	btod/btod_add.C
	btod/btod_contract2_1.C
	btod/btod_contract2_2.C
	btod/btod_contract2_3.C
	btod/btod_contract2_4.C
	btod/btod_contract2_5.C
	btod/btod_contract2_6.C
	btod/btod_copy.C
	btod/btod_sum.C
)

set(SRC_MP
	mp/task_dispatcher.C
	mp/worker.C
	mp/worker_group.C
	mp/worker_pool.C
)

add_library(tensor ${SRC} ${SRC_LINALG} ${SRC_INST} ${SRC_MP})

IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
set_property(SOURCE blas.C APPEND PROPERTY
	COMPILE_FLAGS -msse2)
ENDIF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS LIBTENSOR_INSTANTIATE_TEMPLATES)
set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS LIBTENSOR_TIMINGS)
set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)
