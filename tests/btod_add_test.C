#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libvmm.h>
#include <libtensor.h>
#include "btod_add_test.h"
#include "compare_ref.h"

namespace libtensor {

void btod_add_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	test_1();
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

