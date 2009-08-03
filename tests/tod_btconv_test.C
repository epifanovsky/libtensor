#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor.h>
#include <libvmm.h>
#include "compare_ref.h"
#include "tod_btconv_test.h"

namespace libtensor {

void tod_btconv_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));
	test_1();
	test_2();
	test_3();
}

void tod_btconv_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_btconv_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, default_symmetry<2, double>,
		allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 5);
	bis.split(1, 5);

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	tensor_t t(dims), t_ref(dims);
	tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

	double *pt = tctrl.req_dataptr();
	double *pt_ref = tctrl_ref.req_dataptr();

	//	Fill in random input, generate reference

	size_t sz = dims.get_size();
	for(size_t i = 0; i < sz; i++) {
		pt[i] = drand48();
		pt_ref[i] = 0.0;
	}

	tctrl.ret_dataptr(pt); pt = NULL;
	tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;

	t_ref.set_immutable();
	bt.set_immutable();

	//	Invoke the operation

	tod_btconv<2> op(bt);
	op.perform(t);

	//	Compare the result against the reference

	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

void tod_btconv_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "tod_btconv_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, default_symmetry<2, double>,
		allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 5);
	bis.split(1, 5);

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	tensor_t t(dims), t_ref(dims);
	tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

	double *pt = tctrl.req_dataptr();
	double *pt_ref = tctrl_ref.req_dataptr();

	//	Fill in random input, generate reference

	size_t sz = dims.get_size();
	for(size_t i = 0; i < sz; i++) {
		pt[i] = drand48();
		pt_ref[i] = 0.0;
	}

	index<2> i_00, ii;
	index<2> istart = bis.get_block_start(i_00);
	dimensions<2> dims_00 = bis.get_block_dims(i_00);
	tensor_i<2, double> &blk_00 = btctrl.req_block(i_00);
	tensor_ctrl_t tctrl_00(blk_00);
	double *p_00 = tctrl_00.req_dataptr();
	do {
		index<2> iii(istart);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p_00[dims_00.abs_index(ii)] =
			drand48();
	} while(dims_00.inc_index(ii));
	tctrl_00.ret_dataptr(p_00);
	btctrl.ret_block(i_00);

	tctrl.ret_dataptr(pt); pt = NULL;
	tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;

	t_ref.set_immutable();
	bt.set_immutable();

	//	Invoke the operation

	tod_btconv<2> op(bt);
	op.perform(t);

	//	Compare the result against the reference

	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

void tod_btconv_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "tod_btconv_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, default_symmetry<2, double>,
		allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 5);
	bis.split(1, 5);

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	tensor_t t(dims), t_ref(dims);
	tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

	double *pt = tctrl.req_dataptr();
	double *pt_ref = tctrl_ref.req_dataptr();

	//	Fill in random input, generate reference

	size_t sz = dims.get_size();
	for(size_t i = 0; i < sz; i++) {
		pt[i] = drand48();
		pt_ref[i] = 0.0;
	}

	index<2> i_11, ii;
	i_11[0] = 1; i_11[1] = 1;
	index<2> istart = bis.get_block_start(i_11);
	dimensions<2> dims_11 = bis.get_block_dims(i_11);
	tensor_i<2, double> &blk_11 = btctrl.req_block(i_11);
	tensor_ctrl_t tctrl_11(blk_11);
	double *p_11 = tctrl_11.req_dataptr();
	do {
		index<2> iii(istart);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p_11[dims_11.abs_index(ii)] =
			drand48();
	} while(dims_11.inc_index(ii));
	tctrl_11.ret_dataptr(p_11);
	btctrl.ret_block(i_11);

	tctrl.ret_dataptr(pt); pt = NULL;
	tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;

	t_ref.set_immutable();
	bt.set_immutable();

	//	Invoke the operation

	tod_btconv<2> op(bt);
	op.perform(t);

	//	Compare the result against the reference

	compare_ref<2>::compare(testname, t, t_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

} // namespace libtensor
