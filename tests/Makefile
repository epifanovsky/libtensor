.PHONY: all clean

OBJS = libtensor_suite.o

all: libtensor_tests

clean:
	rm -f *.[ao]
	rm -f libtensor_tests

libtensor_tests: libtensor_tests.o $(OBJS)
	icpc -o libtensor_tests $^ ../../libtest/libtest.a

%.o: %.C; icpc -g -c -I.. -I../../libtest $<

