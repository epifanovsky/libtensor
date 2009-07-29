#include <libvmm.h>
#include <libtensor.h>
#include "btod_sum_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef default_symmetry<2,double> symmetry2;
typedef block_tensor<2,double,symmetry2,allocator> block_tensor2;
/** very basic tests only
**/
void btod_sum_test::perform() throw(libtest::test_exception) {
	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai);
	block_tensor2 bt1(ia), bt2(ai), bt3(ia), bt4(ai);
	permutation<2> p1,p2;
	p2.permute(0,1);

	btod_add<2> add1(p1), add2(p1);
	add1.add_op(bt2,p2,0.5);
	add1.add_op(bt3,p1,0.1);
	add2.add_op(bt4,p1,0.2);
	add2.add_op(bt2,p2,2.0);

	btod_sum<2> sum(add1);
	sum.add_op(add2,0.1);

	sum.perform(bt1);
}

} // namespace libtensor

