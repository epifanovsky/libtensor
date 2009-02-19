#include "tod_set_test.h"
#include "tensor.h"

namespace libtensor {

typedef tensor<double, libvmm::std_allocator<double> > tensor_d;

void tod_set_test::perform() throw(libtest::test_exception) {
	index i1(4), i2(4);
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range ir(i1, i2);
	dimensions dim(ir);
	tensor_d t(dim);

	tod_set op(5.0);
	op.perform(t);
}

} // namespace libtensor

