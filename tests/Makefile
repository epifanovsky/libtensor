.PHONY: all clean

OBJS  = libtensor_suite.o
OBJS += dimensions_test.o
OBJS += index_test.o
OBJS += index_range_test.o
OBJS += tensor_test.o

all: libtensor_tests

include ../Makefile.inc

INCLUDE = -I.. -I../../libtest -I../../libvmm

clean:
	rm -f *.[ao]
	rm -f libtensor_tests

libtensor_tests: libtensor_tests.o $(OBJS)
	$(LD) $(LDFLAGS) -o libtensor_tests $^ ../../libtest/libtest.a ../libtensor.a

dimensions_test.C: dimensions_test.h

index_test.C: index_test.h

index_range_test.C: index_range_test.h

tensor_test.C: tensor_test.h

