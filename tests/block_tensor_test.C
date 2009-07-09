#include <libvmm.h>
#include <libtensor.h>
#include "block_tensor_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;

void block_tensor_test::perform() throw(libtest::test_exception) {
	index<2> i1, i2; i2[0] = 9; i2[1] = 19;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	bis.split(0, 5);
	bis.split(1, 5);
	bis.split(1, 10);
	bis.split(1, 15);
	block_tensor2 bt(bis);

}

} // namespace libtensor
