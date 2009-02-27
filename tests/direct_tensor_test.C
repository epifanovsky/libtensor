#include "direct_tensor_test.h"

namespace libtensor {

typedef direct_tensor<int, libvmm::std_allocator<int> > tensor_int;

void direct_tensor_test::perform() throw(libtest::test_exception) {
	test_ctor();
}

void direct_tensor_test::test_op::perform(tensor_i<int> &t) throw(exception) {
}

void direct_tensor_test::test_ctor() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0]=3; i2[1]=3;
	index_range ir(i1,i2);
	dimensions d(ir);
	test_op op;
	tensor_int dt(op, d);

	tensor_i<int> &dtref(dt);
	dtref.get_dims();
}

} // namespace libtensor

