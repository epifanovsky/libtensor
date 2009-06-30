#include <libvmm.h>
#include <libtensor.h>
#include "btod_copy_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;

void btod_copy_test::perform() throw(libtest::test_exception) {
	bispace<1> i_sp(2), a_sp(3);
	bispace<2> ia(i_sp*a_sp), ai(a_sp*i_sp);
	block_tensor2 bt1(ia), bt2(ai), bt3(ia), bt4(ai);
	permutation<2> p1,p2;
	p2.permute(0,1);

	btod_copy<2> cp1(bt1);
	cp1.perform(bt3);
}

} // namespace libtensor
