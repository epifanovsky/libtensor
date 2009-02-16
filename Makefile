.PHONY: all tests docs

all: tests
 
tests:
	cd tests && $(MAKE)

docs:
	doxygen
	scp -Cpr docs/html/* hogwarts.usc.edu:public_html/libtensor_docs/

index.h: defs.h exception.h

%.o: %.C; g++ -g -c $<

