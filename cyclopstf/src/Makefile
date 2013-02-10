include ../config.mk

DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard shared/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard dist_tensor/*.o))))

include ../rules.mk

test_SUBDIRS = test ctr_comm
test_model_SUBDIRS = test
pgemm_test_SUBDIRS = test
nonsq_pgemm_test_SUBDIRS = test

bench_SUBDIRS = bench ctr_comm
bench_model_SUBDIRS = bench
nonsq_pgemm_bench_SUBDIRS = bench

dist_interface_SUBDIRS = dist_interface

cyclopstf_SUBDIRS = ctr_seq ctr_comm shared dist_tensor 

libtensor_SUBDIRS = libtensor

libscf_SUBDIRS = chem/scf
scf_test_SUBDIRS = chem/scf

autocc_SUBDIRS = chem/cc

ccd_test_SUBDIRS = chem/ccd 
ccd_bench_SUBDIRS = chem/ccd 
ccsd_test_SUBDIRS = chem/ccd 
gemm_test_SUBDIRS = chem/ccd 
ccsd_scf_SUBDIRS = chem/ccd 
ccsd_bench_SUBDIRS = chem/ccd
ccsdt_SUBDIRS = chem/ccsdt

cyclopstf: ${libdir}/libcyclopstf.a
${libdir}/libcyclopstf.a: shared/comm.o \
                          shared/util.o \
                          shared/memcontrol.o \
                          shared/unit_util.o \
                          dist_tensor/cyclopstf.o \
                          dist_tensor/dist_tensor.o \
                          test/unit_test.o 

INCLUDES += -I${top_dir}/src/ctr_comm -I${top_dir}/src/ctr_seq -I${top_dir}/dist_tensor

