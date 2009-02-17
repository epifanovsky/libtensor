CPP = icpc
LD = icpc
CPP_FLAGS = -g -DTENSOR_DEBUG
INCLUDE =

%.o: %.C; $(CPP) $(CPP_FLAGS) $(INCLUDE) -c $<

