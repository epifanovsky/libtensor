OBJS  = libtensor_suite.o
OBJS += bispace_test.o
OBJS += block_tensor_test.o
OBJS += btensor_test.o
OBJS += btod_contract2_test.o
OBJS += contract_test.o
OBJS += contract2_0_4i_test.o
OBJS += contraction2_test.o
OBJS += contraction2_list_test.o
OBJS += contraction2_processor_test.o
OBJS += default_symmetry_test.o
OBJS += dimensions_test.o
OBJS += direct_btensor_test.o
OBJS += direct_tensor_test.o
OBJS += expr_binary_test.o
OBJS += expr_identity_test.o
OBJS += expr_literal_test.o
OBJS += expression_test.o
OBJS += immutable_test.o
OBJS += index_test.o
OBJS += index_range_test.o
OBJS += labeled_btensor_test.o
OBJS += lehmer_code_test.o
OBJS += letter_test.o
OBJS += permutation_test.o
OBJS += permutator_test.o
OBJS += symmetry_test.o
OBJS += tensor_test.o
OBJS += tod_add_test.o
OBJS += tod_compare_test.o
OBJS += tod_contract2_impl_022_test.o
OBJS += tod_contract2_impl_113_test.o
OBJS += tod_contract2_impl_131_test.o
OBJS += tod_contract2_impl_221_test.o
OBJS += tod_contract2_test.o
OBJS += tod_copy_test.o
OBJS += tod_set_test.o
OBJS += tod_sum_test.o
OBJS += tod_symcontract2_test.o

.PHONY: all
all: libtensor_tests

include ../Makefile.inc
INCLUDE += -I.. -I../../libtest -I../../libvmm

.PHONY: clean
clean:
	rm -f *.[ao]
	rm -f libtensor_tests

libtensor_tests: libtensor_tests.o $(OBJS) ../libtensor.a
	$(LD) $(LDFLAGS) libtensor_tests $(LIBS) $^ ../../libtest/libtest.a

default_symmetry_test.C: default_symmetry_test.h

dimensions_test.C: dimensions_test.h

index_test.C: index_test.h

index_range_test.C: index_range_test.h

lehmer_code_test.C: lehmer_code_test.h

permutation_test.C: permutation_test.h

permutator_test.C: permutator_test.h

symmetry_test.C: symmetry_test.h

tensor_test.C: tensor_test.h

tod_set_test.C: tod_set_test.h

