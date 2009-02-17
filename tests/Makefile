.PHONY: all clean

OBJS = libtensor_suite.o

all: libtensor_tests

clean:
	rm -f *.[ao]
	rm -f libtensor_tests

libtensor_tests: libtensor_tests.o $(OBJS)
	g++ -o libtensor_tests $^ ../*.o ../../libtest/libtest.a

%.o: %.C; g++ -g -c -I.. -I../../libtest $<

