.PHONY: all clean

OBJS  = libtensor_suite.o
OBJS += default_symmetry_test.o
OBJS += dimensions_test.o
OBJS += index_test.o
OBJS += index_range_test.o
OBJS += lehmer_code_test.o
OBJS += permutator_test.o
OBJS += tensor_test.o
OBJS += tod_set_test.o

all: ../Makefile.inc libtensor_tests

../Makefile.inc:

include ../Makefile.inc

INCLUDE = -I.. -I../../libtest -I../../libvmm

clean:
	rm -f *.[ao]
	rm -f libtensor_tests

libtensor_tests: libtensor_tests.o $(OBJS)
	$(LD) $(LDFLAGS) $(LIBS) -o libtensor_tests $^ ../../libtest/libtest.a ../libtensor.a

default_symmetry_test.C: default_symmetry_test.h

dimensions_test.C: dimensions_test.h

index_test.C: index_test.h

index_range_test.C: index_range_test.h

lehmer_code_test.C: lehmer_code_test.h

permutator_test.C: permutator_test.h

tensor_test.C: tensor_test.h

tod_set_test.C: tod_set_test.h

