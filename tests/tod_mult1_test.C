#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_mult1.h>
#include "tod_mult1_test.h"
#include "compare_ref.h"

namespace libtensor {


typedef libvmm::std_allocator<double> allocator;


void tod_mult1_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


void tod_mult1_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_mult1_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), ta_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pa_ref[i] = pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb).perform(ta);

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult1_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "tod_mult1_test::test_2()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), ta_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pa_ref[i] = pa[i] / pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb, true).perform(ta);

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult1_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "tod_mult1_test::test_3()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), ta_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pa_ref[i] = pa[i] + pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb).perform(ta, 1.0);

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult1_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "tod_mult1_test::test_4()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), ta_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pa_ref[i] = pa[i] + pa[i] / pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb, true).perform(ta, 1.0);

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult1_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "tod_mult1_test::test_5()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), ta_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pa_ref[i] = pa[i] - 2.0 * pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb).perform(ta, -2.0);

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_mult1_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "tod_mult1_test::test_6()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), ta_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	for(size_t i = 0; i < sz; i++) {
		pa_ref[i] = pa[i] + 0.5 * pa[i] / pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb, true).perform(ta, 0.5);

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
