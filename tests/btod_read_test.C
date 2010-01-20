#include <sstream>
#include <libtensor.h>
#include <libvmm/std_allocator.h>
#include "btod_read_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_read_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


void btod_read_test::test_1() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim) with one block
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
	ctrl.ret_dataptr(p); p = NULL;

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
	//	Block tensor (2-dim), two blocks along each dimension
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
	ctrl.ret_dataptr(p); p = NULL;

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


void btod_read_test::test_3() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim) with one zero block
	//

	static const char *testname = "btod_read_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	tensor<2, double, allocator_t> t(dims), t_ref(dims);
	tod_set<2>().perform(t_ref);

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
	ctrl.ret_dataptr(p); p = NULL;

	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	btod_read<2>(ss).perform(bt);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_read_test::test_4() throw(libtest::test_exception) {

	//
	//	Block tensor (2-dim), two blocks along each dimension,
	//	zero off-diagonal blocks
	//

	static const char *testname = "btod_read_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 2); bis.split(msk2, 3);

	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	index<2> ii;
	btod_random<2> rand;
	rand.perform(bt_ref, ii);
	ii[0] = 1; ii[1] = 1;
	rand.perform(bt_ref, ii);
	tod_btconv<2>(bt_ref).perform(t_ref);

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
	ctrl.ret_dataptr(p); p = NULL;

	btod_read<2>(ss).perform(bt);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_read_test::test_5() throw(libtest::test_exception) {

	//
	//	Block tensor (4-dim) with one block
	//

	static const char *testname = "btod_read_test::test_5()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 4; i2[1] = 5; i2[2] = 4; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	tensor<4, double, allocator_t> t(dims), t_ref(dims);
	tod_random<4>().perform(t_ref);

	std::stringstream ss;
	ss << "4 " << dims[0] << " " << dims[1] << " " << dims[2] << " "
		<< dims[3] << std::endl;

	tensor_ctrl<4, double> ctrl(t_ref);
	const double *p = ctrl.req_const_dataptr();
	abs_index<4> aidx(dims);
	do {
		ss.precision(15);
		ss.setf(std::ios::fixed, std::ios::floatfield);
		ss << p[aidx.get_abs_index()] << " ";
	} while(aidx.inc());
	ctrl.ret_dataptr(p); p = NULL;

	block_index_space<4> bis(dims);
	block_tensor<4, double, allocator_t> bt(bis);
	btod_read<4>(ss).perform(bt);
	tod_btconv<4>(bt).perform(t);

	compare_ref<4>::compare(testname, t, t_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_read_test::test_6() throw(libtest::test_exception) {

	//
	//	Block tensor (4-dim), two blocks along each dimension
	//

	static const char *testname = "btod_read_test::test_6()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 4; i2[1] = 5; i2[2] = 4; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1, m2;
	m1[0] = true; m1[2] = true;
	m2[1] = true; m2[3] = true;
	bis.split(m1, 2); bis.split(m2, 3);

	tensor<4, double, allocator_t> t_ref(dims);
	block_tensor<4, double, allocator_t> bt(bis), bt_ref(bis);
	btod_random<4>().perform(bt_ref);
	tod_btconv<4>(bt_ref).perform(t_ref);

	std::stringstream ss;
	ss << "4 " << dims[0] << " " << dims[1] << " " << dims[2] << " "
		<< dims[3] << std::endl;

	tensor_ctrl<4, double> ctrl(t_ref);
	const double *p = ctrl.req_const_dataptr();
	abs_index<4> aidx(dims);
	do {
		ss.precision(15);
		ss.setf(std::ios::fixed, std::ios::floatfield);
		ss << p[aidx.get_abs_index()] << " ";
	} while(aidx.inc());
	ctrl.ret_dataptr(p); p = NULL;

	btod_read<4>(ss).perform(bt);

	compare_ref<4>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
