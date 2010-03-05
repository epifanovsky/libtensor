#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_mult.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_mult.h>
#include "btod_mult_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_mult_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


/**	\test Elementwise multiplication of two order-2 tensors with no symmetry
		and no zero blocks.
 **/
void btod_mult_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_mult<2>(ta, tb).perform(tc_ref);

	//	Invoke the operation

	btod_mult<2>(bta, btb).perform(btc);
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise multiplication (additive) of two order-2 tensors
		with no symmetry and no zero blocks.
 **/
void btod_mult_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btod_random<2>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_btconv<2>(btc).perform(tc_ref);
	tod_mult<2>(ta, tb).perform(tc_ref, 0.5);

	//	Invoke the operation

	btod_mult<2>(bta, btb).perform(btc, 0.5);
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise division of two order-2 tensors with no symmetry
		and no zero blocks.
 **/
void btod_mult_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_mult<2>(ta, tb, true).perform(tc_ref);

	//	Invoke the operation

	btod_mult<2>(bta, btb, true).perform(btc);
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise division (additive) of two order-2 tensors
		with no symmetry and no zero blocks.
 **/
void btod_mult_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btod_random<2>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_btconv<2>(btc).perform(tc_ref);
	tod_mult<2>(ta, tb, true).perform(tc_ref, 0.5);

	//	Invoke the operation

	btod_mult<2>(bta, btb, true).perform(btc, 0.5);
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
