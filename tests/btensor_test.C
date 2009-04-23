#include <libvmm.h>
#include "btensor_test.h"

namespace libtensor {

void btensor_test::perform() throw(libtest::test_exception) {
	index<4> i1, i2;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	index_range<4> ir(i1,i2);
	dimensions<4> dims(ir);
	btensor<4, double, libvmm::std_allocator<double> > bt(dims);
}

} // namespace libtensor

