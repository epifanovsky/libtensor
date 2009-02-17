.PHONY: all clean tests docs

all: tests

include Makefile.inc

tests:
	cd tests && $(MAKE)

clean:
	cd tests && $(MAKE) clean
 
docs:
	doxygen
	scp -Cpr docs/html/* hogwarts.usc.edu:public_html/libtensor_docs/

index.h: defs.h exception.h

