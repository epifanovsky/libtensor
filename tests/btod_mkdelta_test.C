#include <libtensor.h>
#include "btod_mkdelta_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_mkdelta_test::perform() throw(libtest::test_exception) {

	test_1();
}


void btod_mkdelta_test::test_1() throw(libtest::test_exception) {

	//
	//	Block tensors with just one block
	//

	static const char *testname = "btod_mkdelta_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dimi(index_range<2>(i1, i2));
	i2[0] = 15; i2[1] = 15;
	dimensions<2> dima(index_range<2>(i1, i2));
	i2[0] = 9; i2[1] = 15;
	dimensions<2> dimd(index_range<2>(i1, i2));
	block_index_space<2> bisi(dimi), bisa(dima), bisd(dimd);

	//	Create block tensors and fill them with random data

	block_tensor<2, double, allocator_t> bti(bisi), bta(bisa), btd(bisd);
	btod_random<2>().perform(bti);
	btod_random<2>().perform(bta);
	btod_random<2>().perform(btd);
	bti.set_immutable();
	bta.set_immutable();

	//	Create reference

	tensor<2, double, allocator_t> ti(dimi), ta(dima), td_ref(dimd);
	tod_btconv<2>(bti).perform(ti);
	tod_btconv<2>(bta).perform(ta);
	tod_mkdelta(ti, ta).perform(td_ref);

	//	Invoke operation

	tensor<2, double, allocator_t> td(dimd);
	btod_mkdelta(bti, bta).perform(btd);
	tod_btconv<2>(btd).perform(td);

	//	Compare with reference

	compare_ref<2>::compare(testname, td, td_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
