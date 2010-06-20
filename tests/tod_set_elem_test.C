#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_set_elem.h>
#include "tod_set_elem_test.h"
#include "compare_ref.h"

namespace libtensor {


void tod_set_elem_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
}


void tod_set_elem_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_set_elem_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	tensor<2, double, allocator_t> t(dims), t_ref(dims);

	{
	tensor_ctrl<2, double> tc(t), tc_ref(t_ref);

	//	Fill in random data
	//
	double *d = tc.req_dataptr();
	double *d_ref = tc_ref.req_dataptr();
	size_t sz = dims.get_size();
	for(size_t i = 0; i < sz; i++) d_ref[i] = d[i] = drand48();
	tc.ret_dataptr(d); d = 0;
	tc_ref.ret_dataptr(d_ref); d_ref = 0;

	//	Test [0,0]
	//
	index<2> i00;
	abs_index<2> ai00(i00, dims);
	double q = drand48();
	d_ref = tc_ref.req_dataptr();
	d_ref[ai00.get_abs_index()] = q;
	tc_ref.ret_dataptr(d_ref); d_ref = 0;
	tod_set_elem<2>().perform(t, i00, q);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	//	Test [3, 2]
	//
	index<2> i32; i32[0] = 3; i32[1] = 2;
	abs_index<2> ai32(i32, dims);
	q = drand48();
	d_ref = tc_ref.req_dataptr();
	d_ref[ai32.get_abs_index()] = q;
	tc_ref.ret_dataptr(d_ref); d_ref = 0;
	tod_set_elem<2>().perform(t, i32, q);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
