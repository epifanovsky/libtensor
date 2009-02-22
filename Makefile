.PHONY: all clean tests docs

all: libtensor.a tests

include Makefile.inc

INCLUDE = -I. -I../libvmm

OBJS  = dimensions.o
OBJS += lehmer_code.o
OBJS += tod_set.o

libtensor.a: $(OBJS)
	echo $?
	ar -r libtensor.a $?

tests: libtensor.a
	cd tests && $(MAKE) all

clean:
	rm -f *.[ao]
	cd tests && $(MAKE) clean
 
docs:
	doxygen
	scp -Cpr docs/html/* hogwarts.usc.edu:public_html/libtensor_docs/

dimensions.C: dimensions.h

dimensions.h: defs.h exception.h index.h index_range.h

index.h: defs.h exception.h

tensor_i.h: defs.h exception.h dimensions.h

tod_set.C: tod_set.h

tod_set.h: defs.h exception.h tensor_operation.h

