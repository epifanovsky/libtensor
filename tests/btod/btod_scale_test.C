#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_scale.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include "btod_scale_test.h"
#include "../compare_ref.h"

namespace libtensor {


typedef std_allocator<double> allocator_t;


void btod_scale_test::perform() throw(libtest::test_exception) {

	test_0();
	test_i(3);
	test_i(10);
	test_i(32);

	test_1();
}


template<size_t N>
void btod_scale_test::test_generic(
	const char *testname, block_tensor_i<N, double> &bt, double c)
	throw(libtest::test_exception) {

	try {

	dense_tensor<N, double, allocator_t> t(bt.get_bis().get_dims()),
		t_ref(bt.get_bis().get_dims());
	tod_btconv<N>(bt).perform(t_ref);
	tod_scale<N>(t_ref, c).perform();

	btod_scale<N>(bt, c).perform();
	tod_btconv<N>(bt).perform(t);

	compare_ref<N>::compare(testname, t, t_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Checks that scaling by zero results in all zero blocks
 **/
void btod_scale_test::test_0() throw(libtest::test_exception) {

	static const char *testname = "btod_scale_test::test_0()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 3); bis.split(m, 7);
	dimensions<4> bidims(bis.get_block_index_dims());

	volatile double zero = 0.0;
	volatile double n_one = -1.0;
	volatile double n_zero = n_one * zero;

	block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);
	btod_random<4>().perform(bt1);
	btod_random<4>().perform(bt2);
	test_generic(testname, bt1, zero);
	test_generic(testname, bt2, n_zero);

	block_tensor_ctrl<4, double> ctrl1(bt1);
	abs_index<4> ai1(bidims);
	do {
		if(!ctrl1.req_is_zero_block(ai1.get_index())) {
			fail_test(testname, __FILE__, __LINE__,
				"Bad zero block structure in bt1.");
		}
	} while(ai1.inc());

	block_tensor_ctrl<4, double> ctrl2(bt2);
	abs_index<4> ai2(bidims);
	do {
		if(!ctrl2.req_is_zero_block(ai2.get_index())) {
			fail_test(testname, __FILE__, __LINE__,
				"Bad zero block structure in bt2.");
		}
	} while(ai2.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Checks the scaling and the zero-block structure of one-dim
		tensors
 **/
void btod_scale_test::test_i(size_t i) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "btod_scale_test::test_i(" << i << ")";
	std::string tn = ss.str();

	try {

	index<1> ia, ib;
	ia[0] = i - 1;
	dimensions<1> d(index_range<1>(ia, ib));
	mask<1> m; m[0] = true;

	block_index_space<1> bis1(d);
	block_index_space<1> bis2(d);
	bis2.split(m, i/2);
	block_index_space<1> bis3(d);
	bis3.split(m, i/3); bis3.split(m, 2*i/3);

	//	Test the correct scaling

	block_tensor<1, double, allocator_t> bt1a(bis1);
	block_tensor<1, double, allocator_t> bt2a(bis2);
	block_tensor<1, double, allocator_t> bt3a(bis3);
	btod_random<1>().perform(bt1a);
	btod_random<1>().perform(bt2a);
	btod_random<1>().perform(bt3a);
	test_generic(tn.c_str(), bt1a, 0.5);
	test_generic(tn.c_str(), bt2a, -1.5);
	test_generic(tn.c_str(), bt3a, 2.2);

	index<1> i0, i1, i2;
	i1[0] = 1; i2[0] = 2;

	//	Test the correct zero block structure

	block_tensor<1, double, allocator_t> bt1b(bis1);
	block_tensor<1, double, allocator_t> bt2b(bis2);
	block_tensor<1, double, allocator_t> bt3b(bis3);
	btod_random<1>().perform(bt2b, i1);
	btod_random<1>().perform(bt3b, i0);
	btod_random<1>().perform(bt3b, i2);
	test_generic(tn.c_str(), bt1b, 1.0);
	test_generic(tn.c_str(), bt2b, -0.6);
	test_generic(tn.c_str(), bt3b, -2.7);

	block_tensor_ctrl<1, double> c1b(bt1b), c2b(bt2b), c3b(bt3b);
	if(!c1b.req_is_zero_block(i0)) {
		fail_test(tn.c_str(), __FILE__, __LINE__,
			"!c1b.req_is_zero_block(i0)");
	}
	if(!c2b.req_is_zero_block(i0)) {
		fail_test(tn.c_str(), __FILE__, __LINE__,
			"!c2b.req_is_zero_block(i0)");
	}
	if(c2b.req_is_zero_block(i1)) {
		fail_test(tn.c_str(), __FILE__, __LINE__,
			"c2b.req_is_zero_block(i1)");
	}
	if(c3b.req_is_zero_block(i0)) {
		fail_test(tn.c_str(), __FILE__, __LINE__,
			"c3b.req_is_zero_block(i0)");
	}
	if(!c3b.req_is_zero_block(i1)) {
		fail_test(tn.c_str(), __FILE__, __LINE__,
			"!c3b.req_is_zero_block(i1)");
	}
	if(c3b.req_is_zero_block(i2)) {
		fail_test(tn.c_str(), __FILE__, __LINE__,
			"c3b.req_is_zero_block(i2)");
	}

	} catch(exception &e) {
		fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests proper scaling in block tensors with permutational
		%symmetry and anti-symmetry.
 **/
void btod_scale_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_scale_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 3); bis.split(m, 7);

	block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);

	{
		block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
		se_perm<4, double> elem1(permutation<4>().permute(1, 2), true);
		se_perm<4, double> elem2(permutation<4>().permute(0, 1).
			permute(1, 2).permute(2, 3), true);
		se_perm<4, double> elem3(permutation<4>().permute(0, 1), false);
		se_perm<4, double> elem4(permutation<4>().permute(2, 3), false);
		ctrl1.req_symmetry().insert(elem1);
		ctrl1.req_symmetry().insert(elem2);
		ctrl2.req_symmetry().insert(elem3);
		ctrl2.req_symmetry().insert(elem4);
	}

	btod_random<4>().perform(bt1);
	btod_random<4>().perform(bt2);

	test_generic(testname, bt1, 0.45);
	test_generic(testname, bt2, -1.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
