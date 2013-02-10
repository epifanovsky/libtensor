all: $(DEFAULT_COMPONENTS)

ALL_COMPONENTS = cyclopstf dist_interface examples libtensor test test_model \
                 pgemm_test nonsq_pgemm_test bench bench_model nonsq_pgemm_bench \
                 ccd_bench ccd_test autocc scf_test ccsd_test gemm_test ccsd_bench ccsd_scf \
                 libscf ccsdt

cyclopstf:
dist_interface: autocc cyclopstf
examples: ccd_test scf_test ccsd_test ccsd_scf gemm_test
scf_test ccsd_scf ccsdt: libscf
ccd_test ccd_bench ccsd_test gemm_test ccsd_bench ccsd_scf ccsdt: dist_interface libtensor cyclopstf autocc
scf_test libscf: dist_interface libtensor cyclopstf autocc
test test_model pgemm_test nonsq_pgemm_test: cyclopstf
bench bench_model nonsq_pgemm_bench: cyclopstf

LOWER_NO_UNDERSCORE = 1
LOWER_UNDERSCORE = 2
UPPER_NO_UNDERSCORE = 3
UPPER_UNDERSCORE = 4

F77COMPILE = $(F77) $(_INCLUDES) $(_FFLAGS)
F90COMPILE = $(F90) $(_INCLUDES) $(_FCFLAGS)
CCOMPILE = $(CC) $(_DEFS) $(_INCLUDES) $(_CPPFLAGS) $(_CFLAGS)
CXXCOMPILE = $(CXX) $(_DEFS) $(_INCLUDES) $(_CPPFLAGS) $(_CXXFLAGS)
CCOMPILEDEPS = $(CCOMPILE) $(DEPFLAGS)
CXXCOMPILEDEPS = $(CXXCOMPILE) $(DEPFLAGS)

LINK = $(CXX) $(_CXXFLAGS) $(_LDFLAGS) -o $@
ARCHIVE = $(AR) $@

bindir = ${top_dir}/bin
libdir = ${top_dir}/lib

DEPDIR = .deps
DEPS += ${top_dir}/.dummy $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard *.o))))
ALL_SUBDIRS = $(sort $(SUBDIRS) $(foreach comp,$(ALL_COMPONENTS),$(value $(addsuffix _SUBDIRS,$(comp)))))

_CFLAGS = $(CFLAGS)
_CPPFLAGS = $(CPPFLAGS)
_INCLUDES = $(INCLUDES) -I. -I${top_dir} -I${top_dir}/include
_CXXFLAGS = $(CXXFLAGS)
_DEFS = $(DEFS)
_FFLAGS = $(FFLAGS)
_LDFLAGS = $(LDFLAGS) -L${top_dir}/lib
_DEPENDENCIES = $(DEPENDENCIES) Makefile ${top_dir}/config.mk ${top_dir}/rules.mk
_LIBS = $(LIBS)

.PHONY: all default clean $(ALL_COMPONENTS)
FORCE:

$(ALL_COMPONENTS):
	@for dir in $(SUBDIRS) $($@_SUBDIRS); do \
		echo "Making $@ in $$dir"; \
		(cd $$dir && $(MAKE) $@); \
	done

clean:
	rm -rf $(DEPDIR) *.o
	@for subdir in $(ALL_SUBDIRS); do \
		echo "Making clean in $$subdir"; \
		(cd $$subdir && $(MAKE) clean); \
	done

${bindir}/%: $(_DEPENDENCIES) 
	@rm -f $@
	@mkdir -p $(dir $@)
	$(LINK) $(filter %.o,$^) $(_LIBS)

${libdir}/%: $(_DEPENDENCIES)
	@rm -f $@
	@mkdir -p $(dir $@)
	$(ARCHIVE) $(filter %.o,$^)

%.o: %.f $(_DEPENDENCIES)
	$(F77COMPILE) -c -o $@ $<

%.o: %.f90 $(_DEPENDENCIES)
	$(F90COMPILE) -c -o $@ $<

%.o: %.c $(_DEPENDENCIES)
	@mkdir -p $(DEPDIR)
	$(CCOMPILEDEPS) -c -o $@ $<

%.o: %.cxx $(_DEPENDENCIES)
	@mkdir -p $(DEPDIR)
	$(CXXCOMPILEDEPS) -c -o $@ $<

-include $(DEPS)
