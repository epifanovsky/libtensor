#include <libvmm.h>
#include <libtensor.h>
#include "block_tensor_test.h"

namespace libtensor {

void block_tensor_test::perform() throw(libtest::test_exception) {

	test_orbits_1();
}

void block_tensor_test::test_orbits_1() throw(libtest::test_exception) {

	static const char *testname = "block_tensor_test::test_orbits_1()";

	typedef block_tensor<2, double, default_symmetry<2, double>,
		libvmm::std_allocator<double> > block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;
	typedef orbit_iterator<2, double> orbit_iterator_t;

	try {

	index<2> i0, i1, i2;
	i2[0] = 4; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt(bis);
	block_tensor_ctrl_t ctrl(bt);
	orbit_iterator_t oi = ctrl.req_orbits();

	if(!oi.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expecting an empty block set for a new block tensor.");
	}

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

} // namespace libtensor
