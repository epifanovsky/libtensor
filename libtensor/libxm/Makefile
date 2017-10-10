CC= cc
CFLAGS= -Wall -Wextra -g -fopenmp -Isrc
LDFLAGS= -L/usr/local/lib
LIBS= -lblas -lm

# Intel Compiler (release build)
#CC= icc
#CFLAGS= -DNDEBUG -Wall -Wextra -O3 -fopenmp -mkl=sequential -Isrc
#LDFLAGS=
#LIBS= -lm

# Intel Compiler with MPI (release build)
#CC= mpicc
#CFLAGS= -DXM_USE_MPI -DNDEBUG -Wall -Wextra -O3 -fopenmp -mkl=sequential -Isrc
#LDFLAGS=
#LIBS= -lm

EXAMPLE= example
EXAMPLE_O= example.o
TEST= test
TEST_O= test.o

XM_A= src/libxm.a

all: $(EXAMPLE) $(TEST)

$(EXAMPLE): $(XM_A) $(EXAMPLE_O)
	$(CC) -o $@ $(CFLAGS) $(EXAMPLE_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(TEST): $(XM_A) $(TEST_O)
	$(CC) -o $@ $(CFLAGS) $(TEST_O) $(XM_A) $(LDFLAGS) $(LIBS)

$(XM_A):
	cd src && CC="$(CC)" CFLAGS="$(CFLAGS)" $(MAKE)

check: $(TEST)
	./$(TEST)

checkmpi: $(TEST)
	mpirun -np 3 ./$(TEST)
	mpirun -np 2 ./$(TEST)
	mpirun -np 1 ./$(TEST)

dist:
	git archive --format=tar.gz --prefix=libxm/ -o libxm.tgz HEAD

clean:
	cd src && $(MAKE) clean
	rm -f $(EXAMPLE) $(EXAMPLE_O) $(TEST) $(TEST_O)
	rm -f *.core xmpagefile libxm.tgz

.PHONY: all check checkmpi clean dist
