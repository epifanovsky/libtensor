.PHONY: all
all: deps libtensor_tests

OBJS  = libtensor_suite.o
OBJS += bispace_test.o
OBJS += block_index_space_test.o
OBJS += block_map_test.o
OBJS += block_tensor_test.o
OBJS += btensor_test.o
OBJS += btod_add_test.o
OBJS += btod_compare_test.o
OBJS += btod_contract2_test.o
OBJS += btod_copy_test.o
OBJS += btod_random_test.o
OBJS += btod_sum_test.o
OBJS += contract_test.o
OBJS += contraction2_test.o
OBJS += dimensions_test.o
OBJS += direct_btensor_test.o
OBJS += direct_tensor_test.o
OBJS += dot_product_test.o
OBJS += global_timings_test.o
OBJS += immutable_test.o
OBJS += index_test.o
OBJS += index_range_test.o
OBJS += labeled_btensor_test.o
OBJS += letter_test.o
OBJS += letter_expr_test.o
OBJS += mask_test.o
OBJS += orbit_test.o
OBJS += orbit_list_test.o
OBJS += permutation_test.o
OBJS += permutation_builder_test.o
OBJS += so_projdown_test.o
OBJS += so_projup_test.o
OBJS += symel_cycleperm_test.o
OBJS += symmetry_test.o
OBJS += symmetry_element_base_test.o
OBJS += tensor_test.o
OBJS += timer_test.o
OBJS += timings_test.o
OBJS += tod_add_test.o
OBJS += tod_btconv_test.o
OBJS += tod_compare_test.o
OBJS += tod_contract2_test.o
OBJS += tod_copy_test.o
OBJS += tod_dotprod_test.o
OBJS += tod_random_test.o
OBJS += tod_set_test.o
OBJS += tod_sum_test.o
OBJS += tod_solve_test.o
OBJS += tod_symcontract2_test.o

DEPS = $(OBJS:.o=.d)

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif

include ../Makefile.inc
INCLUDE += -I.. -I../../libtest -I../../libvmm

.PHONY: deps
deps: $(DEPS)

libtensor_tests: libtensor_tests.o $(OBJS) ../libtensor.a
	$(LD) $(LDFLAGS) libtensor_tests $^ ../../libtest/libtest.a $(LIBS) 

.PHONY: clean
clean:
	$(RM) *.a
	$(RM) *.o
	$(RM) *.d
	$(RM) *.d.*
	$(RM) libtensor_tests
