include ../../config.mk
include ../../rules.mk

pgemm_test: ${bindir}/test/pgemm_test
${bindir}/test/pgemm_test: pgemm_test.o

nonsq_pgemm_test: ${bindir}/test/nonsq_pgemm_test
${bindir}/test/nonsq_pgemm_test: nonsq_pgemm_test.o

# what component?
${bindir}/test/suite_contract: unit_test.o test_sym_kernel.o suite_contract.o \
                               ${libdir}/libcyclopstf.a

test: DEFS += -DSYMMETRY
test: test/test_contract test/test_sum test/test_scale
${bindir}/test/test_scale: unit_test.o unit_scale.o unit_test_scl.o \
                           ${libdir}/libcyclopstf.a
${bindir}/test/test_sum: unit_test.o unit_sum.o test_sym_kernel.o unit_test_sum.o \
                         ${libdir}/libcyclopstf.a
${bindir}/test/test_contract: unit_test.o unit_nosym.o unit_contract.o test_sym_kernel.o unit_test_ctr.o \
                              ${libdir}/libcyclopstf.a

test_model: DEFS += -DMODEL
test_model: test/test_model
${bindir}/test/test_model: unit_test.o unit_model.o test_sym_kernel.o unit_test_ctr.o \
                           ${libdir}/libcyclopstf.a

unit_test.o: FORCE

LIBS += -lcyclopstf
