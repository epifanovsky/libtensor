#include <libvmm.h>
#include <libtensor.h>
#include "btod_contract2_test.h"

namespace libtensor {


void btod_contract2_test::perform() throw(libtest::test_exception) {
	typedef libvmm::std_allocator<double> allocator;
	typedef block_tensor<2,double,allocator> block_tensor2;

	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	i2[0] = 1; i2[1] = 1;
	dimensions<2> dims_ij(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai), ij(dims_ij);

	block_tensor2 bt1(ia), bt2(ai), bt3(ij);
	permutation<2> p;

	contraction2<1,1,1> contr(p);
	contr.contract(1,0);

	btod_contract2<1,1,1> operation(contr,bt1,bt2);
	operation.perform(bt3,0.1);

        operation.perform(bt3);

        test_bis_1();
}


void btod_contract2_test::test_bis_1() throw(libtest::test_exception) {

	static const char *testname = "btod_contract2_test::test_bis_1()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<4, double, allocator_t> block_tensor_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bisa(dims), bisc(dims);
	mask<4> msk, msk1, msk2;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk, 3);
	bisa.split(msk, 5);
	bisc.split(msk1, 3);
	bisc.split(msk1, 5);
	bisc.split(msk2, 3);
	bisc.split(msk2, 5);

	block_index_space<4> bisb(bisa);

	block_tensor_t bta(bisa), btb(bisb);
	contraction2<2, 2, 2> contr;
	contr.contract(0, 2);
	contr.contract(1, 3);

	btod_contract2<2, 2, 2> op(contr, bta, btb);

	if(!op.get_bis().equals(bisc)) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected output block index space.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

