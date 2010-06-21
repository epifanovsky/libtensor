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
	tod/loop_list_add.C
	tod/loop_list_copy.C
	tod/loop_list_mul.C
	tod/loop_list_elem.C
	tod/loop_list_elem1.C
)

set(SRC_INST
	tod/tod_add.C
	tod/tod_contract2.C
	btod/additive_btod.C
	btod/basic_btod.C
	btod/btod_add.C
	btod/btod_contract2.C
	btod/btod_copy.C
	btod/btod_sum.C
)

add_library(tensor ${SRC} ${SRC_INST})

set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS LIBTENSOR_INSTANTIATE_TEMPLATES)
set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)
