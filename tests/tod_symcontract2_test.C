#include "tod_symcontract2_test.h"
#include "tensor.h"

namespace libtensor {

typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;

void tod_symcontract2_test::perform() throw(libtest::test_exception) {
	index<4> i1, i2;
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	tensor4_d t(dim);

}

} // namespace libtensor

