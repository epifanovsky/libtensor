.PHONY: all
all: deps libtensor_performance_tests

OBJS  = libtensor_pt_suite.o
OBJS += performance_test_suite.o
OBJS += test_expressions.o

DEPS = $(OBJS:.o=.d)

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif

include ../Makefile.inc
INCLUDE += -I.. -I../../libtest/src -I../../libvmm/src
CPPFLAGS+= -DLIBTENSOR_TIMINGS

.PHONY: deps
deps: $(DEPS)

libtensor_performance_tests: libtensor_performance_tests.o $(OBJS) ../libtensor.a
	$(LD) $(LDFLAGS) libtensor_performance_tests $^ ../../libtest/build/src/libtest.a $(LIBS) 

.PHONY: clean
clean:
	$(RM) *.a
	$(RM) *.o
	$(RM) *.d
	$(RM) libtensor_performance_tests
