.PHONY: all
all: Makefile.inc deps libtensor.a tests

Makefile.inc:
	@echo "-----------------------------------------------"
	@echo "Use Makefile.inc-sample to create Makefile.inc."
	@echo "-----------------------------------------------"

include Makefile.inc

INCLUDE += -I. -I../libvmm

SVNREV := $(shell svnversion -n .)

OBJS  =
OBJS += contract2_0_4i.o
OBJS += defs.o
OBJS += exception.o
OBJS += permutator.o

DEPS = $(OBJS:.o=.d)

.PHONY: deps
deps: $(DEPS)

ifneq ($(MAKECMDGOALS),clean)
include $(DEPS)
endif

libtensor.a: $(OBJS)
	echo $?
	ar -r libtensor.a $?

.PHONY: tests
tests: Makefile.inc libtensor.a
	cd tests && $(MAKE) all

.PHONY: docs
docs:
	doxygen
	scp -Cpr docs/html/* hogwarts.usc.edu:public_html/libtensor_docs/

.PHONY: trac
trac:
	doxygen
	cp -R docs/html/* /var/lib/trac/libtensor/htdocs/

defs.o: CPPFLAGS += -DLIBTENSOR_SVN_REV='"$(SVNREV)"'

.PHONY: clean
clean:
	$(RM) *.a
	$(RM) *.o
	$(RM) *.d
	cd tests && $(MAKE) clean
 