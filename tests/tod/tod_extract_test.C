#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_extract.h>
#include "../compare_ref.h"
#include "tod_extract_test.h"

namespace libtensor {


void tod_extract_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
}


/**	\test Extract a single matrix row: \f$ b_i = a_{ij} |_{j=2} \f$
 **/
void tod_extract_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_extract_test::test_1()";

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
		index<2> idxa; idxa[0] = i; idxa[1] = 2;
		index<1> idxb; idxb[0] = i;
		abs_index<2> aidxa(idxa, dims2);
		abs_index<1> aidxb(idxb, dims1);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
	}

	mask<2> m; m[0] = true; m[1] = false;
	index<2> idx; idx[0] = 0; idx[1] = 2;
	tod_extract<2, 1>(ta, m, idx).perform(tb);

	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a tensor slice with one index fixed:
		\f$ b_{ij} = a_{ikj} |_{k=0} \f$
 **/
void tod_extract_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "tod_extract_test::test_2()";

	typedef std_allocator<double> allocator;

	try {

	size_t ni = 6, nj = 11, nk = 3;
	index<2> i2a, i2b;
	i2b[0] = ni - 1; i2b[1] = nj - 1;
	index<3> i3a, i3b;
	i3b[0] = ni - 1; i3b[1] = nk - 1; i3b[2] = nj - 1;
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
		index<3> idxa; idxa[0] = i; idxa[1] = 0; idxa[2] = j;
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
	index<3> idx; idx[0] = 0; idx[1] = 0; idx[2] = 0;
	tod_extract<3, 1>(ta, m, idx).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a tensor slice with one index fixed and permutation:
		\f$ b_{ji} = a_{ikj} |_{k=0} \f$
 **/
void tod_extract_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "tod_extract_test::test_3()";

	typedef std_allocator<double> allocator;

	try {

	size_t ni = 6, nj = 11, nk = 3;
	index<2> i2a, i2b;
	i2b[0] = nj - 1; i2b[1] = ni - 1;
	index<3> i3a, i3b;
	i3b[0] = ni - 1; i3b[1] = nk - 1; i3b[2] = nj - 1;
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
		index<3> idxa; idxa[0] = i; idxa[1] = 0; idxa[2] = j;
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

	permutation<2> perm;
	perm.permute(0, 1);

	mask<3> m; m[0] = true; m[1] = false; m[2] = true;
	index<3> idx; idx[0] = 0; idx[1] = 0; idx[2] = 0;
	tod_extract<3, 1>(ta, m,perm ,idx).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_extract_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "tod_extract_test::test_4()";

	typedef std_allocator<double> allocator;

	try {


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void tod_extract_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "tod_extract_test::test_5()";

	typedef std_allocator<double> allocator;

	try {


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
