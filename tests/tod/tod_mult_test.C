#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_mult.h>
#include "../compare_ref.h"
#include "tod_mult_test.h"

namespace libtensor {


typedef libvmm::std_allocator<double> allocator;


void tod_mult_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_pq_pq_1(1, 12, false); test_pq_pq_1(12, 1, false);
	test_pq_pq_1(1, 12, true);  test_pq_pq_1(12, 1, true);
	test_pq_pq_1(10, 10, false); test_pq_pq_1(10, 10, true);
	test_pq_pq_2(1, 12, false, 0.1);
	test_pq_pq_2(12, 1, false, -0.2);
	test_pq_pq_2(10, 10, false, 1.0);
	test_pq_pq_2(1, 12, true, -0.6);
	test_pq_pq_2(12, 1, true, 0.8);
	test_pq_pq_2(10, 10, true, 12.0);
	test_pq_qp(false, 0.1);
	test_pq_qp(true, -1.0);
	test_qp_pq(false, -1.1);
	test_qp_pq(true, 0.4);
	test_qp_qp(false, -1.1);
	test_qp_qp(true, 0.4);
	test_pqrs_qprs(2, 4, 5, 6, false, 0.1);
	test_pqrs_qprs(3, 5, 2, 3, true, -0.1);
	test_pqrs_qprs(2, 3, 1, 1, false, -0.1);
	test_pqrs_qprs(3, 1, 2, 1, true, 0.1);
	test_pqrs_qrps(4, 3, 2, 5, false, 1.0);
	test_pqrs_qrps(2, 6, 4, 3, true, 0.3);
	test_pqrs_qrps(4, 1, 2, 1, false, -0.2);
	test_pqrs_qrps(1, 2, 4, 1, true, -0.6);
	test_pqsr_pqrs(2, 4, 6, 5, false, 1.1);
	test_pqsr_pqrs(3, 2, 5, 4, true, -1.2);
	test_pqsr_pqrs(4, 1, 3, 1, false, 0.3);
	test_pqsr_pqrs(2, 4, 1, 1, true, -0.4);
	test_prsq_qrps(4, 2, 3, 5, false, -0.4);
	test_prsq_qrps(5, 3, 6, 4, true, 0.7);
	test_prsq_qrps(4, 3, 1, 1, false, 0.3);
	test_prsq_qrps(1, 4, 3, 1, true, -0.4);
}

void tod_mult_test::test_pq_pq_1(size_t ni, size_t nj, bool recip)
		throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_pq_pq_1("
			<< ni << ", " << nj << ", " << recip << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	if (recip) {
		for(size_t i = 0; i < sz; i++) {
			pc_ref[i] = pa[i] / pb[i];
		}
	}
	else {
		for(size_t i = 0; i < sz; i++) {
			pc_ref[i] = pa[i] * pb[i];
		}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb, recip).perform(tc);

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}



void tod_mult_test::test_pq_pq_2(
		size_t ni, size_t nj, bool recip, double coeff)
		throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_pq_pq_2("
			<< ni << ", " << nj << ", " << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	if (recip) {
		for(size_t i = 0; i < sz; i++)
			pc_ref[i] = pc[i] + coeff * pa[i] / pb[i];
	}
	else {
		for(size_t i = 0; i < sz; i++)
			pc_ref[i] = pc[i] + coeff * pa[i] * pb[i];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<2>(ta, tb, recip).perform(tc, coeff);

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}



void tod_mult_test::test_pq_qp(bool recip, double coeff)
		throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_3(" << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t dim = dims.get_dim(0);
	if (recip) {
		for(size_t i = 0; i < dim; i++)
		for(size_t j = 0; j < dim; j++)
			pc_ref[i * dim + j] = pc[i * dim + j] +
					coeff * pa[i * dim + j] / pb[j * dim + i];
	}
	else {
		for(size_t i = 0; i < dim; i++)
		for(size_t j = 0; j < dim; j++)
			pc_ref[i * dim + j] = pc[i * dim + j] +
					coeff * pa[i * dim + j] * pb[j * dim + i];
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	permutation<2> pa, pb;
	pb.permute(0, 1);
	tod_mult<2>(ta, pa, tb, pb, recip, coeff).perform(tc, 1.0);

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult_test::test_qp_pq(bool recip, double coeff)
		throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_4(" << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t dim = dims.get_dim(0);
	if (recip) {
		for(size_t i = 0; i < dim; i++)
		for(size_t j = 0; j < dim; j++)
			pc_ref[i * dim + j] = pc[i * dim + j] +
					coeff * pa[j * dim + i] / pb[i * dim + j];
	}
	else {
		for(size_t i = 0; i < dim; i++)
		for(size_t j = 0; j < dim; j++)
			pc_ref[i * dim + j] = pc[i * dim + j] +
					coeff * pa[j * dim + i] * pb[i * dim + j];
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	permutation<2> pa, pb;
	pa.permute(0, 1);
	tod_mult<2>(ta, pa, tb, pb, recip, coeff).perform(tc, 1.0);

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult_test::test_qp_qp(bool recip, double coeff)
		throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_5(" << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	tensor<2, double, allocator> ta(dims), tb(dims), tc(dims), tc_ref(dims);

	{
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t dim = dims.get_dim(0);
	if (recip) {
		for(size_t i = 0; i < dim; i++)
		for(size_t j = 0; j < dim; j++)
			pc_ref[i * dim + j] = pc[i * dim + j] +
					coeff * pa[j * dim + i] / pb[j * dim + i];
	}
	else {
		for(size_t i = 0; i < dim; i++)
		for(size_t j = 0; j < dim; j++)
			pc_ref[i * dim + j] = pc[i * dim + j] +
					coeff * pa[j * dim + i] * pb[j * dim + i];
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	permutation<2> pa, pb;
	pa.permute(0, 1);
	pb.permute(0, 1);
	tod_mult<2>(ta, pa, tb, pb, recip, coeff).perform(tc, 1.0);

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult_test::test_pqrs_qprs(
		size_t ni, size_t nj, size_t nk, size_t nl,
		bool recip, double coeff) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_pqrs_qprs(" << ni << ", " << nj << ", " << nk
			<< ", " << nl << ", " << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
	index_range<4> ir(i1, i2);
	dimensions<4> dima(ir), dimb(ir);

	permutation<4> p1, p2;
	p2.permute(0, 1);
	dimb.permute(p2);
	p2.invert();

	tensor<4, double, allocator> ta(dima), tb(dimb), tc(dima), tc_ref(dima);

	{
	tensor_ctrl<4, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	size_t sz = dima.get_size();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t cnt = 0;
	if (recip) {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = j; i1[1] = i; i1[2] = k; i1[3] = l;
			pc_ref[cnt] = pc[cnt] + coeff * pa[cnt] / pb[dimb.abs_index(i1)];
			cnt++;
		}
	}
	else {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = j; i1[1] = i; i1[2] = k; i1[3] = l;
			pc_ref[cnt] = pc[cnt] + coeff * pa[cnt] * pb[dimb.abs_index(i1)];
			cnt++;
		}
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<4>(ta, p1, tb, p2, recip, coeff).perform(tc, 1.0);

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult_test::test_pqrs_qrps(
		size_t ni, size_t nj, size_t nk, size_t nl,
		bool recip, double coeff) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_pqrs_qrps(" << ni << ", " << nj << ", " << nk
			<< ", " << nl << ", " << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
	index_range<4> ir(i1, i2);
	dimensions<4> dima(ir), dimb(ir);

	permutation<4> p1, p2;
	p2.permute(0, 1).permute(1, 2);
	dimb.permute(p2);
	p2.invert();

	tensor<4, double, allocator> ta(dima), tb(dimb), tc(dima), tc_ref(dima);

	size_t sz = dima.get_size();
	{
	tensor_ctrl<4, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t cnt = 0;
	if (recip) {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0]=j; i1[1]=k; i1[2]=i; i1[3]=l;
			pc_ref[cnt] = pc[cnt] + coeff * pa[cnt] / pb[dimb.abs_index(i1)];
			cnt++;
		}
	}
	else {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0]=j; i1[1]=k; i1[2]=i; i1[3]=l;
			pc_ref[cnt] = pc[cnt] + coeff * pa[cnt] * pb[dimb.abs_index(i1)];
			cnt++;
		}
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<4>(ta, p1, tb, p2, recip, coeff).perform(tc, 1.0);

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult_test::test_pqsr_pqrs(
		size_t ni, size_t nj, size_t nk, size_t nl,
		bool recip, double coeff) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_pqsr_pqrs(" << ni << ", " << nj << ", " << nk
			<< ", " << nl << ", " << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
	index_range<4> ir(i1, i2);
	dimensions<4> dima(ir), dimb(ir);
	permutation<4> p1, p2;
	p1.permute(2, 3);
	dima.permute(p1);
	p1.invert();

	tensor<4, double, allocator> ta(dima), tb(dimb), tc(dimb), tc_ref(dimb);

	{
	tensor_ctrl<4, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	size_t sz = dima.get_size();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t cnt = 0;
	if (recip) {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = i; i1[1] = j; i1[2] = l; i1[3] = k;
			pc_ref[cnt] = pc[cnt] + coeff * pa[dima.abs_index(i1)] / pb[cnt];
			cnt++;
		}

	}
	else {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = i; i1[1] = j; i1[2] = l; i1[3] = k;
			pc_ref[cnt] = pc[cnt] + coeff * pa[dima.abs_index(i1)] * pb[cnt];
			cnt++;
		}
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<4>(ta, p1, tb, p2, recip, coeff).perform(tc, 1.0);

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

void tod_mult_test::test_prsq_qrps(size_t ni, size_t nj, size_t nk, size_t nl,
		bool recip, double coeff) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mult_test::test_prsq_qrps(" << ni << ", " << nj << ", " << nk
			<< ", " << nl << ", " << recip << ", " << coeff << ")";
	std::string tns = tnss.str();

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
	index_range<4> ir(i1, i2);
	dimensions<4> dims(ir), dima(ir), dimb(ir);

	permutation<4> p1, p2;
	p1.permute(1, 2).permute(2, 3);
	p2.permute(0, 1).permute(1, 2);

	dima.permute(p1);
	dimb.permute(p2);

	p1.invert();
	p2.invert();

	tensor<4, double, allocator> ta(dima), tb(dimb), tc(dims), tc_ref(dims);

	{
	tensor_ctrl<4, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pc = tcc.req_dataptr();
	double *pc_ref = tcc_ref.req_dataptr();

	size_t sz = dima.get_size();

	for(size_t i = 0; i < sz; i++) pa[i] = drand48();
	for(size_t i = 0; i < sz; i++) pb[i] = drand48();
	for(size_t i = 0; i < sz; i++) pc[i] = drand48();

	size_t cnt = 0;
	if (recip) {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = i; i1[1] = k; i1[2] = l; i1[3] = j;
			i2[0] = j; i2[1] = k; i2[2] = i; i2[3] = l;
			pc_ref[cnt] = pc[cnt] +
					coeff * pa[dima.abs_index(i1)] / pb[dimb.abs_index(i2)];
			cnt++;
		}
	}
	else {
		for(size_t i = 0; i < ni; i++)
		for(size_t j = 0; j < nj; j++)
		for(size_t k = 0; k < nk; k++)
		for(size_t l = 0; l < nl; l++) {
			i1[0] = i; i1[1] = k; i1[2] = l; i1[3] = j;
			i2[0] = j; i2[1] = k; i2[2] = i; i2[3] = l;
			pc_ref[cnt] = pc[cnt] +
					coeff * pa[dima.abs_index(i1)] * pb[dimb.abs_index(i2)];
			cnt++;
		}
	}
	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcc.ret_dataptr(pc); pc = 0;
	tcc_ref.ret_dataptr(pc_ref); pc_ref = 0;
	}

	ta.set_immutable();
	tb.set_immutable();
	tc_ref.set_immutable();

	tod_mult<4>(ta, p1, tb, p2, recip, coeff).perform(tc, 1.0);

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
