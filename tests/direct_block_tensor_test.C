#include <libtensor/core/block_tensor.h>
#include <libtensor/core/direct_block_tensor.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include "direct_block_tensor_test.h"
#include "compare_ref.h"

namespace libtensor {

void direct_block_tensor_test::perform() throw(libtest::test_exception) {

	test_op_1();
	test_op_2();
}


/**	\test Installs a simple copy operation in a direct block %tensor.
 **/
void direct_block_tensor_test::test_op_1() throw(libtest::test_exception) {

	static const char *testname = "direct_block_tensor_test::test_op_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor<2, double, allocator_t> bta(bis);
	btod_random<2>().perform(bta);
	bta.set_immutable();

	btod_copy<2> op_copy(bta);
	direct_block_tensor<2, double, allocator_t> btb(op_copy);

	tensor<2, double, allocator_t> tc(dims), tc_ref(dims);
	tod_btconv<2>(bta).perform(tc_ref);
	tod_btconv<2>(btb).perform(tc);
	compare_ref<2>::compare(testname, tc, tc_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Installs a simple copy operation in a direct block %tensor (multiple blocks).
 **/
void direct_block_tensor_test::test_op_2() throw(libtest::test_exception) {

	static const char *testname = "direct_block_tensor_test::test_op_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk;
	msk[0]=true; msk[1]=true;
	bis.split(msk,4);

	block_tensor<2, double, allocator_t> bta(bis);
	btod_random<2>().perform(bta);
	block_tensor_ctrl<2,double> ctrl(bta);
	i1[1]=1;
	i2[0]=1; i2[1]=0;
	ctrl.req_zero_block(i1);
	ctrl.req_zero_block(i2);
	bta.set_immutable();

	btod_copy<2> op_copy(bta);
	direct_block_tensor<2, double, allocator_t> btb(op_copy);

	tensor<2, double, allocator_t> tc(dims), tc_ref(dims);
	tod_btconv<2>(bta).perform(tc_ref);
	tod_btconv<2>(btb).perform(tc);
	compare_ref<2>::compare(testname, tc, tc_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

