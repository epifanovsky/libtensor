#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/btod/btod_trace.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_trace.h>
#include "btod_trace_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_trace_test::perform() throw(libtest::test_exception) {

	test_zero_1();
	test_nosym_1();
	test_nosym_1_sp();
	test_nosym_2();
	test_nosym_3();
	test_nosym_4();
	test_nosym_5();
	test_nosym_6();
	test_nosym_7();
	test_permsym_1();
	test_permsym_2();
}


/**	\test Computes the trace of a square matrix: \f$ b_i = a_{ii} \f$
		(all zero blocks)
 **/
void btod_trace_test::test_zero_1() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_zero_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);

	block_tensor<2, double, allocator_t> bta(bis);
	bta.set_immutable();

	//	Prepare the reference
	double d_ref = 0.0;

	//	Invoke the operation
	double d = btod_trace<1>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a square matrix: \f$ d = a_{ii} \f$,
		no symmetry
 **/
void btod_trace_test::test_nosym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);

	block_tensor<2, double, allocator_t> bta(bis);
	tensor<2, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<1>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<1>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a square matrix: \f$ d = a_{ii} \f$,
		no symmetry, sparse blocks
 **/
void btod_trace_test::test_nosym_1_sp() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_1_sp()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);

	block_tensor<2, double, allocator_t> bta(bis);
	tensor<2, double, allocator_t> ta(dims);

	//	Fill in random data
	index<2> i00, i12, i22;
	i12[0] = 1; i12[1] = 2;
	i22[0] = 2; i22[1] = 2;
	btod_random<2>().perform(bta, i00);
	btod_random<2>().perform(bta, i12);
	btod_random<2>().perform(bta, i22);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<1>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<1>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a square matrix (with permutation):
		\f$ d = a_{ii} \f$, no symmetry
 **/
void btod_trace_test::test_nosym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);

	block_tensor<2, double, allocator_t> bta(bis);
	tensor<2, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	permutation<2> perm; perm.permute(0, 1);
	tod_btconv<2>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<1>(ta, perm).calculate();

	//	Invoke the operation
	double d = btod_trace<1>(bta, perm).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor:
		\f$ d = a_{ijij} \f$, no symmetry
 **/
void btod_trace_test::test_nosym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_3()";

	typedef std_allocator<double> allocator_t;

	try {

	size_t ni = 10, nj = 11;

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = ni - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1; m1[0] = true; m1[2] = true;
	mask<4> m2; m2[1] = true; m2[3] = true;
	bis.split(m1, 3); bis.split(m1, 7);
	bis.split(m2, 5);

	block_tensor<4, double, allocator_t> bta(bis);
	tensor<4, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<2>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<2>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor
		(with permutation): \f$ d = a_{iijj} \f$, no symmetry
 **/
void btod_trace_test::test_nosym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_4()";

	typedef std_allocator<double> allocator_t;

	try {

	size_t ni = 10, nj = 11;

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = nj - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1; m1[0] = true; m1[1] = true;
	mask<4> m2; m2[2] = true; m2[3] = true;
	bis.split(m1, 3); bis.split(m1, 7);
	bis.split(m2, 5);

	block_tensor<4, double, allocator_t> bta(bis);
	tensor<4, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	permutation<4> perm; perm.permute(1, 2);
	tod_btconv<4>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<2>(ta, perm).calculate();

	//	Invoke the operation
	double d = btod_trace<2>(bta, perm).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a square matrix: \f$ d = a_{ii} \f$,
		no symmetry, blocks with unity dimensions
 **/
void btod_trace_test::test_nosym_5() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_5()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 1);

	block_tensor<2, double, allocator_t> bta(bis);
	tensor<2, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<1>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<1>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor:
		\f$ d = a_{ijij} \f$, no symmetry, blocks with unity dimensions
 **/
void btod_trace_test::test_nosym_6() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_6()";

	typedef std_allocator<double> allocator_t;

	try {

	size_t ni = 3, nj = 2;

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = ni - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1; m1[0] = true; m1[2] = true;
	mask<4> m2; m2[1] = true; m2[3] = true;
	bis.split(m1, 1);
	bis.split(m2, 1);

	block_tensor<4, double, allocator_t> bta(bis);
	tensor<4, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<2>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<2>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor:
		\f$ d = a_{ijij} \f$, no symmetry, all dimensions are equal
 **/
void btod_trace_test::test_nosym_7() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_nosym_7()";

	typedef std_allocator<double> allocator_t;

	try {

	size_t ni = 10, nj = 10;

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = ni - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 2); bis.split(m, 5); bis.split(m, 7);

	block_tensor<4, double, allocator_t> bta(bis);
	tensor<4, double, allocator_t> ta(dims);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<2>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<2>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a square matrix: \f$ d = a_{ii} \f$,
		perm symmetry
 **/
void btod_trace_test::test_permsym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_permsym_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);

	block_tensor<2, double, allocator_t> bta(bis);
	tensor<2, double, allocator_t> ta(dims);

	//	Set up symmetry
	{
		block_tensor_ctrl<2, double> ctrl(bta);
		se_perm<2, double> elem(permutation<2>().permute(0, 1), true);
		ctrl.req_symmetry().insert(elem);
	}

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<1>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<1>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor:
		\f$ d = a_{ijij} \f$, perm symmetry
 **/
void btod_trace_test::test_permsym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_trace_test::test_permsym_2()";

	typedef std_allocator<double> allocator_t;

	try {

	size_t ni = 10, nj = 11;

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = ni - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1; m1[0] = true; m1[2] = true;
	mask<4> m2; m2[1] = true; m2[3] = true;
	bis.split(m1, 3); bis.split(m1, 7);
	bis.split(m2, 5);

	block_tensor<4, double, allocator_t> bta(bis);
	tensor<4, double, allocator_t> ta(dims);

	//	Set up symmetry
	{
		block_tensor_ctrl<4, double> ctrl(bta);
		se_perm<4, double> elem1(permutation<4>().permute(0, 2), true);
		se_perm<4, double> elem2(permutation<4>().permute(1, 3), true);
		ctrl.req_symmetry().insert(elem1);
		ctrl.req_symmetry().insert(elem2);
	}

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	ta.set_immutable();
	double d_ref = tod_trace<2>(ta).calculate();

	//	Invoke the operation
	double d = btod_trace<2>(bta).calculate();

	//	Compare against the reference
	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
