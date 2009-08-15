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
	test_4();
	test_5();
	test_6();
	test_7();
	test_8();
	test_9();

}


void tod_btconv_test::test_1() throw(libtest::test_exception) {

	//
	//	All zero blocks, no symmetry
	//

	static const char *testname = "tod_btconv_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
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

	//
	//	Block [0,0] is non-zero, no symmetry
	//

	static const char *testname = "tod_btconv_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
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

	//
	//	Block [1,1] is non-zero, no symmetry
	//

	static const char *testname = "tod_btconv_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
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


void tod_btconv_test::test_4() throw(libtest::test_exception) {

	//
	//	Diagonal blocks are non-zero, no symmetry
	//

	static const char *testname = "tod_btconv_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 3);
	bis.split(1, 3);

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

	index<2> i_00, i_11, ii;
	i_11[0] = 1; i_11[1] = 1;
	index<2> istart_00 = bis.get_block_start(i_00);
	index<2> istart_11 = bis.get_block_start(i_11);
	dimensions<2> dims_00 = bis.get_block_dims(i_00);
	dimensions<2> dims_11 = bis.get_block_dims(i_11);
	double *p = NULL;

	tensor_i<2, double> &blk_00 = btctrl.req_block(i_00);
	tensor_ctrl_t tctrl_00(blk_00);
	p = tctrl_00.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_00);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_00.abs_index(ii)] =
			drand48();
	} while(dims_00.inc_index(ii));
	tctrl_00.ret_dataptr(p);
	btctrl.ret_block(i_00);

	tensor_i<2, double> &blk_11 = btctrl.req_block(i_11);
	tensor_ctrl_t tctrl_11(blk_11);
	p = tctrl_11.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_11);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_11.abs_index(ii)] =
			drand48();
	} while(dims_11.inc_index(ii));
	tctrl_11.ret_dataptr(p);
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


void tod_btconv_test::test_5() throw(libtest::test_exception) {

	//
	//	Diagonal blocks are non-zero, permutational symmetry
	//

	static const char *testname = "tod_btconv_test::test_5()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 3);
	bis.split(1, 3);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	mask<2> msk;
	msk[0] = true; msk[1] = true;
	symel_cycleperm<2, double> cycle(msk, bidims);
	btctrl.req_sym_add_element(cycle);

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

	index<2> i_00, i_11, ii;
	i_11[0] = 1; i_11[1] = 1;
	index<2> istart_00 = bis.get_block_start(i_00);
	index<2> istart_11 = bis.get_block_start(i_11);
	dimensions<2> dims_00 = bis.get_block_dims(i_00);
	dimensions<2> dims_11 = bis.get_block_dims(i_11);
	double *p = NULL;

	tensor_i<2, double> &blk_00 = btctrl.req_block(i_00);
	tensor_ctrl_t tctrl_00(blk_00);
	p = tctrl_00.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_00);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_00.abs_index(ii)] =
			drand48();
	} while(dims_00.inc_index(ii));
	tctrl_00.ret_dataptr(p);
	btctrl.ret_block(i_00);

	tensor_i<2, double> &blk_11 = btctrl.req_block(i_11);
	tensor_ctrl_t tctrl_11(blk_11);
	p = tctrl_11.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_11);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_11.abs_index(ii)] =
			drand48();
	} while(dims_11.inc_index(ii));
	tctrl_11.ret_dataptr(p);
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


void tod_btconv_test::test_6() throw(libtest::test_exception) {

	//
	//	Off-diagonal blocks are non-zero, no symmetry
	//

	static const char *testname = "tod_btconv_test::test_6()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 3);
	bis.split(1, 3);
	dimensions<2> bidims = bis.get_block_index_dims();

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

	index<2> i_01, i_10, ii;
	i_01[0] = 0; i_01[1] = 1;
	i_10[0] = 1; i_10[1] = 1;
	index<2> istart_01 = bis.get_block_start(i_01);
	index<2> istart_10 = bis.get_block_start(i_10);
	dimensions<2> dims_01 = bis.get_block_dims(i_01);
	dimensions<2> dims_10 = bis.get_block_dims(i_10);
	double *p = NULL;
	permutation<2> perm; perm.permute(0, 1);

	tensor_i<2, double> &blk_01 = btctrl.req_block(i_01);
	tensor_ctrl_t tctrl_01(blk_01);
	p = tctrl_01.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_01);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_01.abs_index(ii)] =
			drand48();
	} while(dims_01.inc_index(ii));
	tctrl_01.ret_dataptr(p);
	btctrl.ret_block(i_01);

	tensor_i<2, double> &blk_10 = btctrl.req_block(i_10);
	tensor_ctrl_t tctrl_10(blk_10);
	p = tctrl_10.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_10);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_10.abs_index(ii)] =
			drand48();
	} while(dims_10.inc_index(ii));
	tctrl_10.ret_dataptr(p);
	btctrl.ret_block(i_10);

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


void tod_btconv_test::test_7() throw(libtest::test_exception) {

	//
	//	Off-diagonal blocks are non-zero, permutational symmetry
	//

	static const char *testname = "tod_btconv_test::test_7()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 3);
	bis.split(1, 3);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	mask<2> msk;
	msk[0] = true; msk[1] = true;
	symel_cycleperm<2, double> cycle(msk, bidims);
	btctrl.req_sym_add_element(cycle);

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

	index<2> i_01, i_10, ii;
	i_01[0] = 0; i_01[1] = 1;
	i_10[0] = 1; i_10[1] = 0;
	index<2> istart_01 = bis.get_block_start(i_01);
	index<2> istart_10 = bis.get_block_start(i_10);
	dimensions<2> dims_01 = bis.get_block_dims(i_01);
	dimensions<2> dims_10 = bis.get_block_dims(i_10);
	double *p = NULL;
	permutation<2> perm; perm.permute(0, 1);

	tensor_i<2, double> &blk_01 = btctrl.req_block(i_01);
	tensor_ctrl_t tctrl_01(blk_01);
	p = tctrl_01.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii1(istart_01), iii2(istart_10);
		index<2> ii2(ii); ii2.permute(perm);
		for(size_t j = 0; j < 2; j++) {
			iii1[j] += ii[j];
			iii2[j] += ii2[j];
		}
		pt_ref[dims.abs_index(iii1)] = pt_ref[dims.abs_index(iii2)] =
			p[dims_01.abs_index(ii)] = drand48();
	} while(dims_01.inc_index(ii));
	tctrl_01.ret_dataptr(p);
	btctrl.ret_block(i_01);

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


void tod_btconv_test::test_8() throw(libtest::test_exception) {

	//
	//	All blocks are non-zero, permutational symmetry
	//

	static const char *testname = "tod_btconv_test::test_8()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(0, 3);
	bis.split(1, 3);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	mask<2> msk;
	msk[0] = true; msk[1] = true;
	symel_cycleperm<2, double> cycle(msk, bidims);
	btctrl.req_sym_add_element(cycle);

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

	index<2> i_00, i_01, i_10, i_11, ii;
	i_01[0] = 0; i_01[1] = 1;
	i_10[0] = 1; i_10[1] = 0;
	i_11[0] = 1; i_11[1] = 1;
	index<2> istart_00 = bis.get_block_start(i_00);
	index<2> istart_01 = bis.get_block_start(i_01);
	index<2> istart_10 = bis.get_block_start(i_10);
	index<2> istart_11 = bis.get_block_start(i_11);
	dimensions<2> dims_00 = bis.get_block_dims(i_00);
	dimensions<2> dims_01 = bis.get_block_dims(i_01);
	dimensions<2> dims_10 = bis.get_block_dims(i_10);
	dimensions<2> dims_11 = bis.get_block_dims(i_11);
	double *p = NULL;
	permutation<2> perm; perm.permute(0, 1);

	tensor_i<2, double> &blk_00 = btctrl.req_block(i_00);
	tensor_ctrl_t tctrl_00(blk_00);
	p = tctrl_00.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_00);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_00.abs_index(ii)] =
			drand48();
	} while(dims_00.inc_index(ii));
	tctrl_00.ret_dataptr(p);
	btctrl.ret_block(i_00);

	tensor_i<2, double> &blk_01 = btctrl.req_block(i_01);
	tensor_ctrl_t tctrl_01(blk_01);
	p = tctrl_01.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii1(istart_01), iii2(istart_10);
		index<2> ii2(ii); ii2.permute(perm);
		for(size_t j = 0; j < 2; j++) {
			iii1[j] += ii[j];
			iii2[j] += ii2[j];
		}
		pt_ref[dims.abs_index(iii1)] = pt_ref[dims.abs_index(iii2)] =
			p[dims_01.abs_index(ii)] = drand48();
	} while(dims_01.inc_index(ii));
	tctrl_01.ret_dataptr(p);
	btctrl.ret_block(i_01);

	tensor_i<2, double> &blk_11 = btctrl.req_block(i_11);
	tensor_ctrl_t tctrl_11(blk_11);
	p = tctrl_11.req_dataptr();
	ii[0] = 0; ii[1] = 0;
	do {
		index<2> iii(istart_11);
		for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
		pt_ref[dims.abs_index(iii)] = p[dims_11.abs_index(ii)] =
			drand48();
	} while(dims_11.inc_index(ii));
	tctrl_11.ret_dataptr(p);
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


void tod_btconv_test::test_9() throw(libtest::test_exception) {

	//
	//	Fully symmetric four-index tensor, one non-zero block
	//

	static const char *testname = "tod_btconv_test::test_9()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<4, double, allocator_t> tensor_t;
	typedef tensor_ctrl<4, double> tensor_ctrl_t;
	typedef block_tensor<4, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<4, double> block_tensor_ctrl_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(0, 3);
	bis.split(1, 3);
	bis.split(2, 3);
	bis.split(3, 3);
	dimensions<4> bidims = bis.get_block_index_dims();

	block_tensor_t bt(bis);
	block_tensor_ctrl_t btctrl(bt);

	mask<4> msk;
	msk[0] = true; msk[1] = true;
	symel_cycleperm<4, double> cycle1(msk, bidims);
	msk[2] = true; msk[3] = true;
	symel_cycleperm<4, double> cycle2(msk, bidims);
	btctrl.req_sym_add_element(cycle1);
	btctrl.req_sym_add_element(cycle2);

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

	index<4> i_0001, i_0010, i_0100, i_1000;
	i_0001[0] = 0; i_0001[1] = 0; i_0001[2] = 0; i_0001[3] = 1;
	i_0010[0] = 0; i_0010[1] = 0; i_0010[2] = 1; i_0010[3] = 0;
	i_0100[0] = 0; i_0100[1] = 1; i_0100[2] = 0; i_0100[3] = 0;
	i_1000[0] = 1; i_1000[1] = 0; i_1000[2] = 0; i_1000[3] = 0;
	index<4> istart_0001 = bis.get_block_start(i_0001);
	index<4> istart_0010 = bis.get_block_start(i_0010);
	index<4> istart_0100 = bis.get_block_start(i_0100);
	index<4> istart_1000 = bis.get_block_start(i_1000);
	dimensions<4> dims_0001 = bis.get_block_dims(i_0001);
	dimensions<4> dims_0010 = bis.get_block_dims(i_0010);
	dimensions<4> dims_0100 = bis.get_block_dims(i_0100);
	dimensions<4> dims_1000 = bis.get_block_dims(i_1000);
	double *p = NULL;
	permutation<4> perm; perm.permute(0, 1).permute(1, 2).permute(2, 3);

	tensor_i<4, double> &blk_0001 = btctrl.req_block(i_0001);
	tensor_ctrl_t tctrl_0001(blk_0001);
	p = tctrl_0001.req_dataptr();

	index<4> ii;
	do {
		index<4> iii1(istart_0001);
		for(size_t j = 0; j < 2; j++) iii1[j] += ii[j];
		index<4> iii2(iii1); iii2.permute(perm);
		index<4> iii3(iii2); iii3.permute(perm);
		index<4> iii4(iii3); iii4.permute(perm);

		pt_ref[dims.abs_index(iii1)] = pt_ref[dims.abs_index(iii2)] =
			pt_ref[dims.abs_index(iii3)] =
			pt_ref[dims.abs_index(iii4)] =
			p[dims_0001.abs_index(ii)] = drand48();
	} while(dims_0001.inc_index(ii));
	tctrl_0001.ret_dataptr(p);
	btctrl.ret_block(i_0001);

	tctrl.ret_dataptr(pt); pt = NULL;
	tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;

	t_ref.set_immutable();
	bt.set_immutable();

	//	Invoke the operation

	tod_btconv<4> op(bt);
	op.perform(t);

	//	Compare the result against the reference

	compare_ref<4>::compare(testname, t, t_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


} // namespace libtensor
