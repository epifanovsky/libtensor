include ../../config.mk
include ../../rules.mk

dist_interface: ${libdir}/libdist_tensor.a
${libdir}/libdist_tensor.a: dist_tensor.o

INCLUDES += -I${top_dir}/src/libtensor/include -I${top_dir}/src/chem/cc
