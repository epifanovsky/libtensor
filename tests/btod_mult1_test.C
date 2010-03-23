#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_mult1.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_mult1.h>
#include "btod_mult1_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_mult1_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


/**	\test Elementwise multiplication of two order-2 tensors with no symmetry
		and no zero blocks.
 **/
void btod_mult1_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_mult1_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);
	tod_mult1<2>(tb).perform(ta_ref);

	//	Invoke the operation

	btod_mult1<2>(btb).perform(bta);
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise multiplication (additive) of two order-2 tensors
		with no symmetry and no zero blocks.
 **/
void btod_mult1_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_mult1_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);
	tod_mult1<2>(tb).perform(ta_ref, 0.5);

	//	Invoke the operation

	btod_mult1<2>(btb).perform(bta, 0.5);
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise division of two order-2 tensors with no symmetry
		and no zero blocks.
 **/
void btod_mult1_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_mult1_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);
	tod_mult1<2>(tb, true).perform(ta_ref);

	//	Invoke the operation

	btod_mult1<2>(btb, true).perform(bta);
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise division (additive) of two order-2 tensors
		with no symmetry and no zero blocks.
 **/
void btod_mult1_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "btod_mult1_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);
	tod_mult1<2>(tb, true).perform(ta_ref, 0.5);

	//	Invoke the operation

	btod_mult1<2>(btb, true).perform(bta, 0.5);
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
