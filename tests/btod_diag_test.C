#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_diag.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_diag.h>
#include "btod_diag_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_diag_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


/**	\test Extract a single diagonal: \f$ b_i = a_{ii} \f$
 **/
void btod_diag_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	tensor<2, double, allocator_t> ta(dims2);
	tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	tod_diag<2, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);
	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test
 **/
void btod_diag_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
