#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_mult1.h>
#include "../compare_ref.h"
#include "tod_mult1_test.h"

namespace libtensor {


typedef libvmm::std_allocator<double> allocator;


void tod_mult1_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_pq_pq_1(5, 10, false); test_pq_pq_1(5, 10, true);
	test_pq_pq_1(5, 1, false); test_pq_pq_1(1, 5, true);
	test_pq_pq_2(5, 10, false, 0.2); test_pq_pq_2(5, 10, true, -0.5);
	test_pq_pq_2(5, 1, false, -0.1); test_pq_pq_2(1, 5, true, 0.2);
	test_pqrs_qrps(5, 7, 3, 4, false, 0.2);
	test_pqrs_qrps(4, 3, 5, 6, true, -0.7);
	test_pqrs_qrps(6, 7, 1, 1, false, -1.1);
	test_pqrs_qrps(1, 8, 5, 1, true, 0.4);
}

void tod_mult1_test::test_pq_pq_1(
		size_t ni, size_t nj,
		bool recip) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult1_test::test_pq_pq_1(" << ni << ", " << nj << ", "
			<< recip << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1;
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

	if (recip) {
		for(size_t i = 0; i < sz; i++) {
			pa_ref[i] = pa[i] / pb[i];
		}
	}
	else {
		for(size_t i = 0; i < sz; i++) {
			pa_ref[i] = pa[i] * pb[i];
		}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb, recip).perform(ta);

	compare_ref<2>::compare(tns.c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult1_test::test_pq_pq_2(
		size_t ni, size_t nj,
		bool recip, double coeff) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult1_test::test_pq_pq_2(" << ni << ", " << nj << ", "
			<< recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1;
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

	if (recip) {
		for(size_t i = 0; i < sz; i++) {
			pa_ref[i] = pa[i] + coeff * pa[i] / pb[i];
		}
	}
	else {
		for(size_t i = 0; i < sz; i++) {
			pa_ref[i] = pa[i] + coeff * pa[i] * pb[i];
		}

	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<2>(tb, recip, coeff).perform(ta, 1.0);

	compare_ref<2>::compare(tns.c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult1_test::test_pqrs_qrps(
		size_t ni, size_t nj, size_t nk, size_t nl,
		bool recip, double coeff) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult1_test::test_pqrs_qrps(" << ni << ", " << nj << ", "
			<< nk << ", " << nl << ", " << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
	index_range<4> ir(i1, i2);
	dimensions<4> dima(ir), dimb(ir);

	permutation<4> p;
	p.permute(0, 1).permute(1, 2);
	dimb.permute(p);

	p.invert();

	size_t sz = dima.get_size();

	tensor<4, double, allocator> ta(dima), tb(dimb), ta_ref(dima);

	{
	tensor_ctrl<4, double> tca(ta), tcb(tb), tca_ref(ta_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pa_ref = tca_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();

	size_t cnt = 0;
	if (recip) {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = j; i1[1] = k; i1[2] = i; i1[3] = l;
			pa_ref[cnt] = pa[cnt] + coeff * pa[cnt] / pb[dimb.abs_index(i1)];
			cnt++;
		}
	}
	else {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = j; i1[1] = k; i1[2] = i; i1[3] = l;
			pa_ref[cnt] = pa[cnt] + coeff * pa[cnt] * pb[dimb.abs_index(i1)];
			cnt++;
		}
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tca_ref.ret_dataptr(pa_ref); pa_ref = 0;
	}

	tb.set_immutable();
	ta_ref.set_immutable();

	tod_mult1<4>(tb, p, recip, coeff).perform(ta, 1.0);

	compare_ref<4>::compare(tns.c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
