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
)

set(SRC_LINALG
	linalg/linalg_impl_generic.C
	linalg/linalg_impl_cblas.C
	linalg/linalg_impl_acml.C
	linalg/linalg_impl_mkl.C
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
