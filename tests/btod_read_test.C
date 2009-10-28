#include <sstream>
#include <libtensor.h>
#include <libvmm.h>
#include "btod_read_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_read_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


void btod_read_test::test_1() throw(libtest::test_exception) {

	//
	//	Block tensor with one block
	//

	static const char *testname = "btod_read_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	tensor<2, double, allocator_t> t(dims), t_ref(dims);
	tod_random<2>().perform(t_ref);

	std::stringstream ss;
	ss << "2 " << dims[0] << " " << dims[1] << std::endl;

	tensor_ctrl<2, double> ctrl(t_ref);
	const double *p = ctrl.req_const_dataptr();
	for(size_t i = 0; i < dims[0]; i++) {
		index<2> idx;
		idx[0] = i;
		for(size_t j = 0; j < dims[1]; j++) {
			idx[1] = j;
			abs_index<2> aidx(idx, dims);
			ss.precision(15);
			ss.setf(std::ios::fixed, std::ios::floatfield);
			ss << p[aidx.get_abs_index()] << " ";
		}
		ss << std::endl;
	}

	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis);
	btod_read<2>(ss).perform(bt);
	tod_btconv<2>(bt).perform(t);

	compare_ref<2>::compare(testname, t, t_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_read_test::test_2() throw(libtest::test_exception) {

	//
	//	Block tensor, two blocks along each dimension
	//

	static const char *testname = "btod_read_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	tensor<2, double, allocator_t> t(dims), t_ref(dims);
	tod_random<2>().perform(t_ref);

	std::stringstream ss;
	ss << "2 " << dims[0] << " " << dims[1] << std::endl;

	tensor_ctrl<2, double> ctrl(t_ref);
	const double *p = ctrl.req_const_dataptr();
	for(size_t i = 0; i < dims[0]; i++) {
		index<2> idx;
		idx[0] = i;
		for(size_t j = 0; j < dims[1]; j++) {
			idx[1] = j;
			abs_index<2> aidx(idx, dims);
			ss.precision(15);
			ss.setf(std::ios::fixed, std::ios::floatfield);
			ss << p[aidx.get_abs_index()] << " ";
		}
		ss << std::endl;
	}

	block_index_space<2> bis(dims);
	mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 2); bis.split(msk2, 3);
	block_tensor<2, double, allocator_t> bt(bis);
	btod_read<2>(ss).perform(bt);
	tod_btconv<2>(bt).perform(t);

	compare_ref<2>::compare(testname, t, t_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
