#include <libvmm.h>
#include <libtensor.h>
#include "btod_contract2_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef default_symmetry<2,double> symmetry2;
typedef block_tensor<2,double,symmetry2,allocator> block_tensor2;


void btod_contract2_test::perform() throw(libtest::test_exception) {
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

}

} // namespace libtensor

