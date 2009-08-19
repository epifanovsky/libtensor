#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libvmm.h>
#include <libtensor.h>
#include "btod_add_test.h"
#include "compare_ref.h"

namespace libtensor {

void btod_add_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	test_1();
	test_2(2.0, 1.0);
	test_2(1.0, 1.0);
	test_3(2.0, 1.0);
	test_3(1.0, 1.0);
	test_exc();
}

void btod_add_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_add_test::test_1()";

	typedef index<2> index_t;
	typedef index_range<2> index_range_t;
	typedef dimensions<2> dimensions_t;
	typedef permutation<2> permutation_t;
	typedef block_index_space<2> block_index_space_t;
	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor_i<2, double> block_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index_t i0, i1, i2;
	i2[0] = 10; i2[1] = 20;
	dimensions_t dims_ia(index_range_t(i1, i2));
	i2[0] = 20; i2[1] = 10;
	dimensions_t dims_ai(index_range_t(i1, i2));
	block_index_space_t bis_ia(dims_ia), bis_ai(dims_ai);

	permutation_t perma1, perma2;
	perma2.permute(0, 1);
	double ca1 = 1.0, ca2 = 0.5;

	block_tensor_t bta1(bis_ia), bta2(bis_ai), btb(bis_ia), btb_ref(bis_ia);
	block_tensor_ctrl_t ctrl_bta1(bta1), ctrl_bta2(bta2), ctrl_btb(btb),
		ctrl_btb_ref(btb_ref);

	tensor_ctrl_t ctrl_blka1(ctrl_bta1.req_block(i0));
	tensor_ctrl_t ctrl_blka2(ctrl_bta2.req_block(i0));
	tensor_ctrl_t ctrl_blkb(ctrl_btb.req_block(i0));
	tensor_ctrl_t ctrl_blkb_ref(ctrl_btb_ref.req_block(i0));
	double *ptr_a1 = ctrl_blka1.req_dataptr();
	double *ptr_a2 = ctrl_blka2.req_dataptr();
	double *ptr_b = ctrl_blkb.req_dataptr();
	double *ptr_b_ref = ctrl_blkb_ref.req_dataptr();

	//	Fill in random input and prepare the reference

	index<2> ia;
	do {
		index<2> ai(ia);
		ai.permute(perma2);
		size_t abs_ia = dims_ia.abs_index(ia);
		size_t abs_ai = dims_ai.abs_index(ai);
		ptr_a1[abs_ia] = drand48();
		ptr_a2[abs_ai] = drand48();
		ptr_b[abs_ia] = drand48();
		ptr_b_ref[abs_ia] = ca1 * ptr_a1[abs_ia] + ca2 * ptr_a2[abs_ai];
	} while(dims_ia.inc_index(ia));

	ctrl_blka1.ret_dataptr(ptr_a1); ptr_a1 = NULL;
	ctrl_blka2.ret_dataptr(ptr_a2); ptr_a2 = NULL;
	ctrl_blkb.ret_dataptr(ptr_b); ptr_b = NULL;
	ctrl_blkb_ref.ret_dataptr(ptr_b_ref); ptr_b_ref = NULL;

	ctrl_bta1.ret_block(i0);
	ctrl_bta2.ret_block(i0);
	ctrl_btb.ret_block(i0);
	ctrl_btb_ref.ret_block(i0);

	//	Run the addition operation

	btod_add<2> op(bta1, perma1, ca1);
	op.add_op(bta2, perma2, ca2);
	op.perform(btb);

	//	Compare against the reference

	tensor_t tb(dims_ia), tb_ref(dims_ia);
	tod_btconv<2> convb(btb), convb_ref(btb_ref);
	convb.perform(tb);
	convb_ref.perform(tb_ref);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void btod_add_test::test_2(double ca1, double ca2)
	throw(libtest::test_exception) {

	//
	//	Arg 1: Non-zero off-diagonal blocks, permutational symmetry
	//	Arg 2: Non-zero diagonal blocks, permutational symmetry
	//

	std::ostringstream tnss;
	tnss << "btod_add_test::test_2(" << ca1 << ", " << ca2 << ")";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor_i<2, double> block_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims_00(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 6;
	dimensions<2> dims_01(index_range<2>(i1, i2));
	i2[0] = 6; i2[1] = 6;
	dimensions<2> dims_11(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> splmsk; splmsk[0] = true; splmsk[1] = true;
	bis.split(splmsk, 3);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor_t bta1(bis), bta2(bis), btb(bis);
	block_tensor_ctrl_t ctrl_bta1(bta1), ctrl_bta2(bta2), ctrl_btb(btb);

	mask<2> msk;
	msk[0] = true; msk[1] = true;
	symel_cycleperm<2, double> cycle(msk, dims);
	ctrl_bta1.req_sym_add_element(cycle);
	ctrl_bta2.req_sym_add_element(cycle);
	ctrl_btb.req_sym_add_element(cycle);

	index<2> i_00, i_01, i_11;
	i_01[0] = 0; i_01[1] = 1;
	i_11[0] = 1; i_11[1] = 1;

	//	Fill in random input

	double *ptr_a1, *ptr_a2, *ptr_b;
	size_t sz;

	tensor_ctrl_t ctrl_blka1_01(ctrl_bta1.req_block(i_01));
	tensor_ctrl_t ctrl_blkb_01(ctrl_btb.req_block(i_01));

	ptr_a1 = ctrl_blka1_01.req_dataptr();
	ptr_b = ctrl_blkb_01.req_dataptr();
	sz = dims_01.get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr_a1[i] = drand48();
		ptr_b[i] = drand48();
	}
	ctrl_blka1_01.ret_dataptr(ptr_a1);
	ctrl_blkb_01.ret_dataptr(ptr_b);

	ctrl_bta1.ret_block(i_01);
	ctrl_btb.ret_block(i_01);

	tensor_ctrl_t ctrl_blka2_00(ctrl_bta1.req_block(i_00));
	tensor_ctrl_t ctrl_blkb_00(ctrl_btb.req_block(i_00));

	ptr_a2 = ctrl_blka2_00.req_dataptr();
	ptr_b = ctrl_blkb_00.req_dataptr();
	sz = dims_00.get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr_a2[i] = drand48();
		ptr_b[i] = drand48();
	}
	ctrl_blka2_00.ret_dataptr(ptr_a2);
	ctrl_blkb_00.ret_dataptr(ptr_b);

	ctrl_bta2.ret_block(i_00);
	ctrl_btb.ret_block(i_00);

	tensor_ctrl_t ctrl_blka2_11(ctrl_bta1.req_block(i_11));
	tensor_ctrl_t ctrl_blkb_11(ctrl_btb.req_block(i_11));

	ptr_a2 = ctrl_blka2_11.req_dataptr();
	ptr_b = ctrl_blkb_11.req_dataptr();
	sz = dims_11.get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr_a2[i] = drand48();
		ptr_b[i] = drand48();
	}
	ctrl_blka2_11.ret_dataptr(ptr_a2);
	ctrl_blkb_11.ret_dataptr(ptr_b);

	ctrl_bta2.ret_block(i_11);
	ctrl_btb.ret_block(i_11);

	//	 Prepare the reference

	tensor_t ta1(dims), ta2(dims), tb_ref(dims);
	tod_btconv<2> conva1(bta1);
	conva1.perform(ta1);
	tod_btconv<2> conva2(bta2);
	conva2.perform(ta2);

//	permutation<2> perm0;
	tod_add<2> op_ref(ta1, ca1);
//	op_ref.add_op(ta1, perm0, ca1);
	op_ref.add_op(ta2, ca2);
	op_ref.perform(tb_ref);

	//	Run the addition operation

	btod_add<2> op(bta1, ca1);
	op.add_op(bta2, ca2);
	op.perform(btb);

	tensor_t tb(dims);
	tod_btconv<2> convb(btb);
	convb.perform(tb);

	//	Compare against the reference

	compare_ref<2>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


void btod_add_test::test_3(double ca1, double ca2)
	throw(libtest::test_exception) {

	//
	//	Arg 1: Non-zero off-diagonal blocks, permutational symmetry
	//	Arg 2: Non-zero diagonal blocks, permutational symmetry
	//

	std::ostringstream tnss;
	tnss << "btod_add_test::test_3(" << ca1 << ", " << ca2 << ")";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor_i<4, double> block_t;
	typedef tensor<4, double, allocator_t> tensor_t;
	typedef tensor_ctrl<4, double> tensor_ctrl_t;
	typedef block_tensor<4, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<4, double> block_tensor_ctrl_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims_0000(index_range<4>(i1, i2));
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 6;
	dimensions<4> dims_0001(index_range<4>(i1, i2));
	i2[0] = 6; i2[1] = 6; i2[2] = 6; i2[3] = 6;
	dimensions<4> dims_1111(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> splmsk;
	splmsk[0] = true; splmsk[1] = true; splmsk[2] = true; splmsk[3] = true;
	bis.split(splmsk, 3);
	dimensions<4> bidims(bis.get_block_index_dims());

	block_tensor_t bta1(bis), bta2(bis), btb(bis);
	block_tensor_ctrl_t ctrl_bta1(bta1), ctrl_bta2(bta2), ctrl_btb(btb);

	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	symel_cycleperm<4, double> cycle1(msk, dims);
	msk[2] = false; msk[3] = false;
	symel_cycleperm<4, double> cycle2(msk, dims);
	ctrl_bta1.req_sym_add_element(cycle1);
	ctrl_bta1.req_sym_add_element(cycle2);
	ctrl_bta2.req_sym_add_element(cycle1);
	ctrl_bta2.req_sym_add_element(cycle2);
	ctrl_btb.req_sym_add_element(cycle1);
	ctrl_btb.req_sym_add_element(cycle2);

	index<4> i_0000, i_0001, i_1111;
	i_0001[0] = 0; i_0001[1] = 0; i_0001[2] = 0; i_0001[3] = 1;
	i_1111[0] = 1; i_1111[1] = 1; i_1111[2] = 1; i_1111[3] = 1;

	//	Fill in random input

	double *ptr_a1, *ptr_a2, *ptr_b;
	size_t sz;

	tensor_ctrl_t ctrl_blka1_0001(ctrl_bta1.req_block(i_0001));
	tensor_ctrl_t ctrl_blkb_0001(ctrl_btb.req_block(i_0001));

	ptr_a1 = ctrl_blka1_0001.req_dataptr();
	ptr_b = ctrl_blkb_0001.req_dataptr();
	sz = dims_0001.get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr_a1[i] = drand48();
		ptr_b[i] = drand48();
	}
	ctrl_blka1_0001.ret_dataptr(ptr_a1);
	ctrl_blkb_0001.ret_dataptr(ptr_b);

	ctrl_bta1.ret_block(i_0001);
	ctrl_btb.ret_block(i_0001);

	tensor_ctrl_t ctrl_blka2_0000(ctrl_bta1.req_block(i_0000));
	tensor_ctrl_t ctrl_blkb_0000(ctrl_btb.req_block(i_0000));

	ptr_a2 = ctrl_blka2_0000.req_dataptr();
	ptr_b = ctrl_blkb_0000.req_dataptr();
	sz = dims_0000.get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr_a2[i] = drand48();
		ptr_b[i] = drand48();
	}
	ctrl_blka2_0000.ret_dataptr(ptr_a2);
	ctrl_blkb_0000.ret_dataptr(ptr_b);

	ctrl_bta2.ret_block(i_0000);
	ctrl_btb.ret_block(i_0000);

	tensor_ctrl_t ctrl_blka2_1111(ctrl_bta1.req_block(i_1111));
	tensor_ctrl_t ctrl_blkb_1111(ctrl_btb.req_block(i_1111));

	ptr_a2 = ctrl_blka2_1111.req_dataptr();
	ptr_b = ctrl_blkb_1111.req_dataptr();
	sz = dims_1111.get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr_a2[i] = drand48();
		ptr_b[i] = drand48();
	}
	ctrl_blka2_1111.ret_dataptr(ptr_a2);
	ctrl_blkb_1111.ret_dataptr(ptr_b);

	ctrl_bta2.ret_block(i_1111);
	ctrl_btb.ret_block(i_1111);

	bta1.set_immutable();
	bta2.set_immutable();

	//	 Prepare the reference

	tensor_t ta1(dims), ta2(dims), tb_ref(dims);
	tod_btconv<4> conva1(bta1);
	conva1.perform(ta1);
	tod_btconv<4> conva2(bta2);
	conva2.perform(ta2);

//	permutation<4> perm0;
	tod_add<4> op_ref(ta1,ca1);
//	op_ref.add_op(ta1, perm0, ca1);
	op_ref.add_op(ta2, ca2);
	op_ref.perform(tb_ref);

	//	Run the addition operation

	btod_add<4> op(bta1, ca1);
	op.add_op(bta2, ca2);
	op.perform(btb);

	tensor_t tb(dims);
	tod_btconv<4> convb(btb);
	convb.perform(tb);

	//	Compare against the reference

	compare_ref<4>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


void btod_add_test::test_exc() throw(libtest::test_exception) {
	/*
	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai);

	block_tensor2 bt1(ia), bt2(ai);
	permutation<2> p1,p2;
	p1.permute(0,1);

	btod_add<2> add(p1);

	bool ok=false;
	try {
		add.add_op(bt1,p2,0.5);
		add.add_op(bt2,p2,1.0);
	}
	catch(exception e) {
		ok=true;
	}

	if(!ok) {
		fail_test("btod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous operands");
	}

	ok=false;
	try {
		add.add_op(bt2,p1,1.0);
		add.perform(bt1);
	}
	catch(exception e) {
		ok=true;
	}

	if(!ok) {
		fail_test("btod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous result tensor");
	}
*/
}


} // namespace libtensor

