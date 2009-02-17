CPP = icpc
LD = icpc
CPP_FLAGS = -g -DDEBUG_CHECK
LD_FLAGS = -g
INCLUDE =

%.o: %.C; $(CPP) $(CPP_FLAGS) $(INCLUDE) -c $<

