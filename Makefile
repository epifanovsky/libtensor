.PHONY: all
all: Makefile.inc libtensor.a tests

Makefile.inc:
	@echo "-----------------------------------------------"
	@echo "Use Makefile.inc-sample to create Makefile.inc."
	@echo "-----------------------------------------------"

include Makefile.inc

INCLUDE += -I. -I../libvmm

SVNREV := $(shell svnversion -n .)

OBJS  =
OBJS += contract2_0_4i.o
OBJS += contract2_2_2i.o
OBJS += contract2_2_3i.o
OBJS += contract2_4_1i.o
OBJS += defs.o
OBJS += exception.o
OBJS += expression_builder.o
OBJS += permutator.o

libtensor.a: $(OBJS)
	echo $?
	ar -r libtensor.a $?

.PHONY: tests
tests: Makefile.inc libtensor.a
	cd tests && $(MAKE) all

.PHONY: clean
clean:
	rm -f *.[ao]
	cd tests && $(MAKE) clean
 
.PHONY: docs
docs:
	doxygen
	scp -Cpr docs/html/* hogwarts.usc.edu:public_html/libtensor_docs/

.PHONY: trac
trac:
	doxygen
	cp -R docs/html/* /var/lib/trac/libtensor/htdocs/

defs.o: CPPFLAGS += -DLIBTENSOR_SVN_REV='"$(SVNREV)"'

dimensions.C: dimensions.h

dimensions.h: defs.h exception.h index.h index_range.h

index.h: defs.h exception.h

lehmer_code.C: lehmer_code.h

tensor_i.h: defs.h exception.h dimensions.h

tod_set.C: tod_set.h

tod_set.h: defs.h exception.h 

