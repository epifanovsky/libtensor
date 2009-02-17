.PHONY: all clean

OBJS = libtensor_suite.o

all: libtensor_tests

include ../Makefile.inc
INCLUDE = -I.. -I../../libtest

clean:
	rm -f *.[ao]
	rm -f libtensor_tests

libtensor_tests: libtensor_tests.o $(OBJS)
	$(LD) -o libtensor_tests $^ ../../libtest/libtest.a

