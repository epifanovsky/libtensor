CPP = icpc
LD = icpc
CPPFLAGS = -g -DDEBUG_CHECK 
LDFLAGS = -g
INCLUDE =

%.o: %.C; $(CPP) $(CPPFLAGS) $(INCLUDE) -c $<

