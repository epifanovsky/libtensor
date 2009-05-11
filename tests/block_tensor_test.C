#include <libvmm.h>
#include "block_tensor_test.h"
#include "bispace.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;

void block_tensor_test::perform() throw(libtest::test_exception) {
	bispace<1> i_sp(10), a_sp(20);
	i_sp.split(5); a_sp.split(5).split(10).split(15);
	bispace<2> ia(i_sp*a_sp);
	block_tensor2 bt(ia);

}

} // namespace libtensor
