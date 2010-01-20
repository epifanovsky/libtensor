#include <libtensor.h>
#include <libvmm/std_allocator.h>
#include "block_map_test.h"

namespace libtensor {

void block_map_test::perform() throw(libtest::test_exception) {

	test_create();
	test_immutable();
}

void block_map_test::test_create() throw(libtest::test_exception) {

	static const char *testname = "block_map_test::test_create()";

	try {

	typedef libvmm::std_allocator<double> allocator_t;
	block_map<2, double, allocator_t> map;

	if(map.contains(0)) {
		fail_test(testname, __FILE__, __LINE__,
			"Nonexisting block 0 reported to be found (1)");
	}
	if(map.contains(2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Nonexisting block 2 reported to be found (1)");
	}

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 5;
	dimensions<2> dims1(index_range<2>(i1, i2));
	i2[0] = 4; i2[1] = 6;
	dimensions<2> dims2(index_range<2>(i1, i2));

	map.create(0, dims1);
	if(!map.contains(0)) {
		fail_test(testname, __FILE__, __LINE__,
			"Existing block 0 cannot be found (2)");
	}
	if(map.contains(2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Nonexisting block 2 reported to be found (2)");
	}

	tensor_i<2, double> &ta1 = map.get(0);
	if(!ta1.get_dims().equals(dims1)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block 0 has incorrect dimensions (2)");
	}

	map.create(2, dims2);
	if(!map.contains(0)) {
		fail_test(testname, __FILE__, __LINE__,
			"Existing block 0 cannot be found (3)");
	}
	if(!map.contains(2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Existing block 2 cannot be found (3)");
	}

	tensor_i<2, double> &tb1 = map.get(0);
	tensor_i<2, double> &tb2 = map.get(2);
	if(!tb1.get_dims().equals(dims1)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block 0 has incorrect dimensions (3)");
	}
	if(!tb2.get_dims().equals(dims2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block 2 has incorrect dimensions (3)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void block_map_test::test_immutable() throw(libtest::test_exception) {

	static const char *testname = "block_map_test::test_immutable()";

	try {

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	block_map<2, double, allocator_t> map;

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 5;
	dimensions<2> dims1(index_range<2>(i1, i2));
	i2[0] = 4; i2[1] = 6;
	dimensions<2> dims2(index_range<2>(i1, i2));

	map.create(0, dims1);
	map.create(2, dims2);

	map.set_immutable();

	tensor_t &tb1 = map.get(0);
	tensor_t &tb2 = map.get(2);

	if(!tb1.is_immutable()) {
		fail_test(testname, __FILE__, __LINE__,
			"Block 0 is not immutable.");
	}
	if(!tb2.is_immutable()) {
		fail_test(testname, __FILE__, __LINE__,
			"Block 2 is not immutable.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
