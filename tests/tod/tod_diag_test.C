#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/tod/tod_diag.h>
#include "../compare_ref.h"
#include "tod_diag_test.h"

namespace libtensor {


void tod_diag_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


/**	\test Extract a single diagonal: \f$ b_i = a_{ii} \f$
 **/
void tod_diag_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_1()";

	typedef std_allocator<double> allocator;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	size_t sza = dims2.get_size(), szb = dims1.get_size();

	dense_tensor<2, double, allocator> ta(dims2);
	dense_tensor<1, double, allocator> tb(dims1), tb_ref(dims1);

	{
	dense_tensor_ctrl<2, double> tca(ta);
	dense_tensor_ctrl<1, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < szb; i++) {
		index<2> idxa; idxa[0] = i; idxa[1] = i;
		index<1> idxb; idxb[0] = i;
		abs_index<2> aidxa(idxa, dims2);
		abs_index<1> aidxb(idxb, dims1);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<2> m; m[0] = true; m[1] = true;
	tod_diag<2, 2>(ta, m).perform(tb);

	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a single diagonal with one index intact:
		\f$ b_{ij} = a_{iij} \f$
 **/
void tod_diag_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_2()";

    typedef std_allocator<double> allocator;

	try {

	size_t ni = 6, nj = 11;
	index<2> i2a, i2b;
	i2b[0] = ni - 1; i2b[1] = nj - 1;
	index<3> i3a, i3b;
	i3b[0] = ni - 1; i3b[1] = ni - 1; i3b[2] = nj - 1;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	size_t sza = dims3.get_size(), szb = dims2.get_size();

	dense_tensor<3, double, allocator> ta(dims3);
	dense_tensor<2, double, allocator> tb(dims2), tb_ref(dims2);

	{
	dense_tensor_ctrl<3, double> tca(ta);
	dense_tensor_ctrl<2, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
		index<3> idxa; idxa[0] = i; idxa[1] = i; idxa[2] = j;
		index<2> idxb; idxb[0] = i; idxb[1] = j;
		abs_index<3> aidxa(idxa, dims3);
		abs_index<2> aidxb(idxb, dims2);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<3> m; m[0] = true; m[1] = true; m[2] = false;
	tod_diag<3, 2>(ta, m).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a single diagonal with one index intact:
		\f$ b_{ij} = a_{iji} \f$
 **/
void tod_diag_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_3()";

    typedef std_allocator<double> allocator;

	try {

	size_t ni = 6, nj = 11;
	index<2> i2a, i2b;
	i2b[0] = ni - 1; i2b[1] = nj - 1;
	index<3> i3a, i3b;
	i3b[0] = ni - 1; i3b[1] = nj - 1; i3b[2] = ni - 1;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	size_t sza = dims3.get_size(), szb = dims2.get_size();

	dense_tensor<3, double, allocator> ta(dims3);
	dense_tensor<2, double, allocator> tb(dims2), tb_ref(dims2);

	{
	dense_tensor_ctrl<3, double> tca(ta);
	dense_tensor_ctrl<2, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
		index<3> idxa; idxa[0] = i; idxa[1] = j; idxa[2] = i;
		index<2> idxb; idxb[0] = i; idxb[1] = j;
		abs_index<3> aidxa(idxa, dims3);
		abs_index<2> aidxb(idxb, dims2);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<3> m; m[0] = true; m[1] = false; m[2] = true;
	tod_diag<3, 2>(ta, m).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a single diagonal with one index intact:
		\f$ b_{ji} = a_{jii} \f$
 **/
void tod_diag_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_4()";

    typedef std_allocator<double> allocator;

	try {

	size_t ni = 6, nj = 11;
	index<2> i2a, i2b;
	i2b[0] = nj - 1; i2b[1] = ni - 1;
	index<3> i3a, i3b;
	i3b[0] = nj - 1; i3b[1] = ni - 1; i3b[2] = ni - 1;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	size_t sza = dims3.get_size(), szb = dims2.get_size();

	dense_tensor<3, double, allocator> ta(dims3);
	dense_tensor<2, double, allocator> tb(dims2), tb_ref(dims2);

	{
	dense_tensor_ctrl<3, double> tca(ta);
	dense_tensor_ctrl<2, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
		index<3> idxa; idxa[0] = j; idxa[1] = i; idxa[2] = i;
		index<2> idxb; idxb[0] = j; idxb[1] = i;
		abs_index<3> aidxa(idxa, dims3);
		abs_index<2> aidxb(idxb, dims2);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<3> m; m[0] = false; m[1] = true; m[2] = true;
	tod_diag<3, 2>(ta, m).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a single diagonal with one index intact and permuted:
		output \f$ b_{ij} = a_{jii} \f$
 **/
void tod_diag_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_5()";

    typedef std_allocator<double> allocator;

	try {

	size_t ni = 6, nj = 11;
	index<2> i2a, i2b;
	i2b[0] = ni - 1; i2b[1] = nj - 1;
	index<3> i3a, i3b;
	i3b[0] = nj - 1; i3b[1] = ni - 1; i3b[2] = ni - 1;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	size_t sza = dims3.get_size(), szb = dims2.get_size();

	dense_tensor<3, double, allocator> ta(dims3);
	dense_tensor<2, double, allocator> tb(dims2), tb_ref(dims2);

	{
	dense_tensor_ctrl<3, double> tca(ta);
	dense_tensor_ctrl<2, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
		index<3> idxa; idxa[0] = j; idxa[1] = i; idxa[2] = i;
		index<2> idxb; idxb[0] = i; idxb[1] = j;
		abs_index<3> aidxa(idxa, dims3);
		abs_index<2> aidxb(idxb, dims2);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<3> m; m[0] = false; m[1] = true; m[2] = true;
	permutation<2> permb; permb.permute(0, 1);
	tod_diag<3, 2>(ta, m, permb).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract a single diagonal with one index intact and permuted:
		output \f$ b_{jik} = a_{ikjk} \f$
 **/
void tod_diag_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_6()";

    typedef std_allocator<double> allocator;

	try {

	size_t ni = 2, nj = 3, nk = 5;
	index<3> i3a, i3b;
	i3b[0] = nj - 1; i3b[1] = ni - 1; i3b[2] = nk - 1;
	index<4> i4a, i4b;
	i4b[0] = ni - 1; i4b[1] = nk - 1; i4b[2] = nj - 1; i4b[3] = nk - 1;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	size_t sza = dims4.get_size(), szb = dims3.get_size();

	dense_tensor<4, double, allocator> ta(dims4);
	dense_tensor<3, double, allocator> tb(dims3), tb_ref(dims3);

	{
	dense_tensor_ctrl<4, double> tca(ta);
	dense_tensor_ctrl<3, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
		index<4> idxa; idxa[0] = i; idxa[1] = k; idxa[2] = j; idxa[3] = k;
		index<3> idxb; idxb[0] = j; idxb[1] = i; idxb[2] = k;
		abs_index<4> aidxa(idxa, dims4);
		abs_index<3> aidxb(idxb, dims3);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}
	}
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<4> m; m[0] = false; m[1] = true; m[2] = false; m[3] = true;
	permutation<3> permb; permb.permute(0, 1).permute(0, 2);
	tod_diag<4, 2>(ta, m, permb).perform(tb);

	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
