.PHONY: all clean tests docs

all: tests

include Makefile.inc
INCLUDE = -I. -I../libvmm

tests:
	cd tests && $(MAKE) all

clean:
	cd tests && $(MAKE) clean
 
docs:
	doxygen
	scp -Cpr docs/html/* hogwarts.usc.edu:public_html/libtensor_docs/

index.h: defs.h exception.h

