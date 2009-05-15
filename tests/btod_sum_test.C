#include <libvmm.h>
#include <libtensor.h>
#include "btod_sum_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;
/** very basic tests only
**/
void btod_sum_test::perform() throw(libtest::test_exception) {
	bispace<1> i_sp(2), a_sp(3);
	bispace<2> ia(i_sp*a_sp), ai(a_sp*i_sp);
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

