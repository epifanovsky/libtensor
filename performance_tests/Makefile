.PHONY: all
all: deps libtensor_performance_tests

OBJS  = libtensor_performance_suite.o
OBJS += expression_p_test.o

DEPS = $(OBJS:.o=.d)

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif

include ../Makefile.inc
INCLUDE += -I.. -I../../libtest -I../../libvmm

.PHONY: deps
deps: $(DEPS)

libtensor_performance_tests: libtensor_performance_tests.o $(OBJS) ../libtensor.a
	$(LD) $(LDFLAGS) libtensor_performance_tests $^ ../../libtest/libtest.a $(LIBS) 

.PHONY: clean
clean:
	$(RM) *.a
	$(RM) *.o
	$(RM) *.d
	$(RM) libtensor_performance_tests
