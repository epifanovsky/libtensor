set(SRC
	backtrace.C
	blas.C
	defs.C
	exception.C
	global_timings.C
	timer.C
	version.C
	tod/loop_list_add.C
	tod/loop_list_copy.C
)
add_library(tensor ${SRC})

set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
set_property(TARGET tensor APPEND PROPERTY
	COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)

