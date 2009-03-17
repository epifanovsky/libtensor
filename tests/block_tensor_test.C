#include <libvmm.h>
#include "block_tensor_test.h"

namespace libtensor {

void block_tensor_test::perform() throw(libtest::test_exception) {
	index i1(4), i2(4);
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	index_range ir(i1,i2);
	dimensions dims(ir);
	block_tensor<double, libvmm::std_allocator<double> > bt(dims);
}

} // namespace libtensor

