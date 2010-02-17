#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_mult.h>
#include "tod_mult_test.h"
#include "compare_ref.h"

namespace libtensor {


typedef libvmm::std_allocator<double> allocator;


void tod_mult_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


void tod_mult_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_mult_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pc_ref[i] = pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb).perform(tc);

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "tod_mult_test::test_2()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pc_ref[i] = pa[i] / pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb, true).perform(tc);

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "tod_mult_test::test_3()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pc_ref[i] = pc[i] + pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb).perform(tc, 1.0);

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "tod_mult_test::test_4()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pc_ref[i] = pc[i] + pa[i] / pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb, true).perform(tc, 1.0);

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "tod_mult_test::test_5()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pc_ref[i] = pc[i] - 2.0 * pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb).perform(tc, -2.0);

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "tod_mult_test::test_6()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pc_ref[i] = pc[i] + 0.5 * pa[i] / pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb, true).perform(tc, 0.5);

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
