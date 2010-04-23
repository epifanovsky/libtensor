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


/**	\test Extract a single diagonal: \f$ b_{ija} = a_{iajb} \f$
 **/
void btod_diag_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<3> bis3(dims3);
	block_index_space<4> bis4(dims4);

	mask<3> msk3;
	msk3[2] = true;
	bis3.split(msk3, 6);
	mask<4> msk4;
	msk4[1] = true; msk4[3]=true;
	bis4.split(msk4, 6);

	block_tensor<4, double, allocator_t> bta(bis4);
	block_tensor<3, double, allocator_t> btb(bis3);

	tensor<4, double, allocator_t> ta(dims4);
	tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	permutation<3> pb;
	pb.permute(1,2);

	mask<4> msk;
	msk[1] = true; msk[3] = true;

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	tod_diag<4, 2>(ta, msk, pb).perform(tb_ref);

	//	Invoke the operation
	btod_diag<4, 2>(bta, msk, pb).perform(btb);
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
