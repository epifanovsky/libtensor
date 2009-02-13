.PHONY: all tests

all: tests
 
tests:
	cd tests && $(MAKE)

index.h: defs.h exception.h

%.o: %.C; g++ -g -c $<

