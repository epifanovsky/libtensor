XM_A= libxm.a
XM_O= alloc.o \
      blockspace.o \
      contract.o \
      dim.o \
      scalar.o \
      tensor.o \
      util.o \
      xm.o

AR= ar rc
RANLIB= ranlib

$(XM_A): $(XM_O)
	$(AR) $@ $(XM_O)
	$(RANLIB) $@

clean:
	rm -f $(XM_A) $(XM_O)

.PHONY: clean
