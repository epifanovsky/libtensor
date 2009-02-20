CPP = icpc
LD = icpc
CPPFLAGS = -g -DDEBUG_CHECK 
LDFLAGS = -g
INCLUDE =
LIBS =

%.o: %.C; $(CPP) $(CPPFLAGS) $(INCLUDE) -c $<

