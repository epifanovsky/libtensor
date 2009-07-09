#include <libvmm.h>
#include <libtensor.h>
#include "btod_copy_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;

void btod_copy_test::perform() throw(libtest::test_exception) {
	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai);

	block_tensor2 bt1(ia), bt2(ai), bt3(ia), bt4(ai);
	permutation<2> p1,p2;
	p2.permute(0,1);

	btod_copy<2> cp1(bt1);
	cp1.perform(bt3);
}

} // namespace libtensor
