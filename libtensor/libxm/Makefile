# Scalar type is double
SCALAR_TYPE= XM_SCALAR_DOUBLE
# Scalar type is float
#SCALAR_TYPE= XM_SCALAR_FLOAT
# Scalar type is double complex
#SCALAR_TYPE= XM_SCALAR_DOUBLE_COMPLEX
# Scalar type is float complex
#SCALAR_TYPE= XM_SCALAR_FLOAT_COMPLEX

# Clang with Netlib BLAS on FreeBSD (debug build)
CC= clang
CFLAGS= -D$(SCALAR_TYPE) -Weverything -Wno-gnu-imaginary-constant -Wno-padded -Wno-used-but-marked-unused -Wno-missing-noreturn -Wno-format-nonliteral -fcolor-diagnostics -g -DHAVE_ARC4RANDOM -DHAVE_BITSTRING_H -DHAVE_TREE_H
LDFLAGS= -L/usr/local/lib -L/usr/local/lib/gcc48
LIBS= -lblas -lgfortran -lpthread -lm

# Intel Compiler with MKL on Linux (release build)
#CC= icc
#CFLAGS= -D$(SCALAR_TYPE) -DNDEBUG -Wall -Wextra -O3 -I./compat -mkl=parallel
#LDFLAGS=
#LIBS= -lpthread -lm

# Intel Compiler with CUDA on Linux (release build)
#CC= icc
#CFLAGS= -D$(SCALAR_TYPE) -DNDEBUG -Wall -Wextra -O3 -I./compat
#LDFLAGS= -L/usr/usc/cuda/default/lib64
#LIBS= -lnvblas -lpthread -lm

# GNU gcc with Netlib BLAS on Linux (debug build)
#CC= gcc
#CFLAGS= -D$(SCALAR_TYPE) -Wall -Wextra -g -I./compat
#LDFLAGS=
#LIBS= -lblas -lpthread -lm

# Clang with Netlib BLAS on OpenBSD (debug build)
#CC= clang
#CFLAGS= -D$(SCALAR_TYPE) -Weverything -Wno-padded -Wno-used-but-marked-unused -Wno-missing-noreturn -Wno-format-nonliteral -fcolor-diagnostics -g -DHAVE_ARC4RANDOM -DHAVE_BITSTRING_H -DHAVE_TREE_H
#LDFLAGS= -L/usr/local/lib
#LIBS= -lblas -lg2c -lpthread -lm

BENCHMARK= benchmark
BENCHMARK_O= benchmark.o
TEST= test
TEST_O= test.o
TEST3= test3
TEST3_O= test3.o

AUX_O= aux.o
XM_A= xm.a
XM_O= alloc.o xm.o

AR= ar rcu
RANLIB= ranlib
RM= rm -f

all: $(BENCHMARK) $(TEST) $(TEST3)

$(BENCHMARK): $(AUX_O) $(XM_A) $(BENCHMARK_O)
	$(CC) -o $@ $(CFLAGS) $(BENCHMARK_O) $(AUX_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(TEST): $(AUX_O) $(XM_A) $(TEST_O)
	$(CC) -o $@ $(CFLAGS) $(TEST_O) $(AUX_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(TEST3): $(AUX_O) $(XM_A) $(TEST3_O)
	$(CC) -o $@ $(CFLAGS) $(TEST3_O) $(AUX_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(XM_A): $(XM_O)
	$(AR) $@ $(XM_O)
	$(RANLIB) $@

check: $(TEST) $(TEST3)
	@./test3 2>/dev/null
	@./test 30 2>/dev/null

dist:
	git archive --format=tar.gz --prefix=libxm/ -o libxm.tgz HEAD

clean:
	$(RM) $(XM_A) $(XM_O) $(AUX_O)
	$(RM) $(BENCHMARK) $(BENCHMARK_O)
	$(RM) $(TEST) $(TEST_O)
	$(RM) $(TEST3) $(TEST3_O)
	$(RM) *.core mapping libxm.tgz

.PHONY: all check clean dist
