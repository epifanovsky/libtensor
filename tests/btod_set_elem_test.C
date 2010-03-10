#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/tensor.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_set_elem.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_set_elem.h>
#include "btod_set_elem_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_set_elem_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


void btod_set_elem_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_set_elem_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t(dims), t_ref(dims);

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	//	Test [0,0] in [0,0]
	//
	index<2> i00;
	btod_set_elem<2>().perform(bt, i00, i00, 2.0);
	tod_set_elem<2>().perform(t_ref, i00, 2.0);
	tod_btconv<2>(bt).perform(t);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	//	Test [3,2] in [0,0]
	//
	index<2> i32; i32[0] = 3; i32[1] = 2;
	btod_set_elem<2>().perform(bt, i00, i32, -2.0);
	tod_set_elem<2>().perform(t_ref, i32, -2.0);
	tod_btconv<2>(bt).perform(t);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_set_elem_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_set_elem_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10;
	m01[1] = true; m10[0] = true;
	bis.split(m10, 3);
	bis.split(m01, 4);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t(dims), t_ref(dims);

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	//	Test element [0,0] in block [0,0]
	//
	index<2> i00;
	btod_set_elem<2>().perform(bt, i00, i00, 2.0);
	tod_set_elem<2>().perform(t_ref, i00, 2.0);
	tod_btconv<2>(bt).perform(t);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	//	Test element [1,2] in block [0,0]
	//
	index<2> i12; i12[0] = 1; i12[1] = 2;
	btod_set_elem<2>().perform(bt, i00, i12, -2.0);
	tod_set_elem<2>().perform(t_ref, i12, -2.0);
	tod_btconv<2>(bt).perform(t);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	//	Test element [0,1] in block [1,0]
	//
	index<2> i01, i10, i31;
	i01[1] = 1; i10[0] = 1;
	i31[0] = 3; i31[1] = 1;
	btod_set_elem<2>().perform(bt, i10, i01, 1.5);
	tod_set_elem<2>().perform(t_ref, i31, 1.5);
	tod_btconv<2>(bt).perform(t);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	//	Test element [1,2] in block [1,1]
	//
	index<2> i11, i46;
	i11[0] = 1; i11[1] = 1;
	i46[0] = 4; i46[1] = 6;
	btod_set_elem<2>().perform(bt, i11, i12, -0.3);
	tod_set_elem<2>().perform(t_ref, i46, -0.3);
	tod_btconv<2>(bt).perform(t);
	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
