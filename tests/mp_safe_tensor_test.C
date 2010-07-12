#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor_ctrl.h>
#include "mp_safe_tensor_test.h"

namespace libtensor {


void mp_safe_tensor_test::perform() throw(libtest::test_exception) {

	test_1();
}


void mp_safe_tensor_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "mp_safe_tensor_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sz = dims.get_size();

	mp_safe_tensor<2, double, allocator_t> t1(dims);
	tensor_ctrl<2, double> c1(t1);

	c1.req_prefetch();

	double *p1 = c1.req_dataptr();
	for(size_t i = 0; i < sz; i++) p1[i] = (double)i;
	c1.ret_dataptr(p1); p1 = 0;

	const double *p2 = c1.req_const_dataptr();
	for(size_t i = 0; i < sz; i++) {
		if(p2[i] != (double)i) {
			fail_test(testname, __FILE__, __LINE__,
				"Data corruption detected.");
		}
	}
	c1.ret_const_dataptr(p2);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
