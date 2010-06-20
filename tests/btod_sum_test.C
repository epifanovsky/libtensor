#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_contract2.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_sum.h>
#include "btod_sum_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_sum_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6(true);
	test_6(false);
}


void btod_sum_test::test_1() throw(libtest::test_exception) {

	//
	//	Single operand A + B
	//

	static const char *testname = "btod_sum_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt3_ref(bis);
	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();


	btod_add<2> add(bt1);
	add.add_op(bt2, 2.0);
	add.add_op(bt2);

	btod_sum<2> sum(add);
	sum.perform(bt3);
	add.perform(bt3_ref);

	compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_sum_test::test_2() throw(libtest::test_exception) {

	//
	//	Two operands: A + B and C + D
	//

	static const char *testname = "btod_sum_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis),
		bt5(bis), bt5_ref(bis);
	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_random<2>().perform(bt4);
	btod_random<2>().perform(bt5);
	btod_copy<2>(bt5).perform(bt5_ref);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();

	btod_add<2> add1(bt1), add2(bt3), add_ref(bt1);
	add1.add_op(bt2);
	add2.add_op(bt4);
	add_ref.add_op(bt2);
	add_ref.add_op(bt3);
	add_ref.add_op(bt4);

	btod_sum<2> sum(add1);
	sum.add_op(add2);
	sum.perform(bt5);
	add_ref.perform(bt5_ref);

	compare_ref<2>::compare(testname, bt5, bt5_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_sum_test::test_3() throw(libtest::test_exception) {

	//
	//	Two operands: A + B and C + D
	//

	static const char *testname = "btod_sum_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis),
		bt5(bis), bt5_ref(bis);
	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_random<2>().perform(bt4);
	bt1.set_immutable();
	bt2.set_immutable();
	bt3.set_immutable();
	bt4.set_immutable();

	btod_add<2> add1(bt1), add2(bt3), add_ref(bt1);
	add1.add_op(bt2);
	add2.add_op(bt4);
	add_ref.add_op(bt2);
	add_ref.add_op(bt3, -1.0);
	add_ref.add_op(bt4, -1.0);

	btod_sum<2> sum(add1);
	sum.add_op(add2, -1.0);
	sum.perform(bt5);
	add_ref.perform(bt5_ref);

	compare_ref<2>::compare(testname, bt5, bt5_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_sum_test::test_4() throw(libtest::test_exception) {

	//
	//	Two operands: A and C + D
	//

	static const char *testname = "btod_sum_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<4, double, allocator_t> block_tensor_t;

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 10; i2[2] = 5; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1, m2;
	m1[0] = true; m1[2] = true;
	m2[1] = true; m2[3] = true;
	bis.split(m1, 2);
	bis.split(m2, 3);
	bis.split(m2, 6);

	block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis), bt4_ref(bis);
	btod_random<4>().perform(bt1);
	btod_random<4>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();

	permutation<4> perm;
	perm.permute(1, 3);
	btod_add<4> add1(bt1), add2(bt2), add_ref(bt1);
	add2.add_op(bt2, perm, -1.0);
	btod_copy<4>(bt2, perm, -1.0).perform(bt3);
	bt3.set_immutable();
	add_ref.add_op(bt2);
	add_ref.add_op(bt3);

	btod_sum<4> sum(add1);
	sum.add_op(add2);
	sum.perform(bt4);
	add_ref.perform(bt4_ref);

	compare_ref<4>::compare(testname, bt4, bt4_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_sum_test::test_5() throw(libtest::test_exception) {

	//
	//	Single operand A * B
	//

	static const char *testname = "btod_sum_test::test_5()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<4> i1, i2;
	i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
	dimensions<4> dims_iiaa(index_range<4>(i1, i2));
	i2[0] = 12; i2[1] = 6; i2[2] = 6; i2[3] = 6;
	dimensions<4> dims_iaaa(index_range<4>(i1, i2));
	block_index_space<4> bis_iiaa(dims_iiaa), bis_iaaa(dims_iaaa);
	mask<4> m1, m2, m3, m4;
	m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
	m3[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis_iiaa.split(m1, 3);
	bis_iiaa.split(m1, 7);
	bis_iiaa.split(m1, 10);
	bis_iiaa.split(m2, 2);
	bis_iiaa.split(m2, 3);
	bis_iiaa.split(m2, 5);
	bis_iaaa.split(m3, 3);
	bis_iaaa.split(m3, 7);
	bis_iaaa.split(m3, 10);
	bis_iaaa.split(m4, 2);
	bis_iaaa.split(m4, 3);
	bis_iaaa.split(m4, 5);

	block_tensor<4, double, allocator_t> bta(bis_iaaa);
	block_tensor<4, double, allocator_t> btb(bis_iiaa);
	block_tensor<4, double, allocator_t> btc(bis_iaaa), btc_ref(bis_iaaa);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Run contraction and compute the reference

	//	iabc = kcad ikbd
	//	caib->iabc
	contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2).
		permute(2, 3));
	contr.contract(0, 1);
	contr.contract(3, 3);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc_ref);

	btod_sum<4> sum(op);
	sum.perform(btc);

	compare_ref<4>::compare(testname, btc, btc_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void btod_sum_test::test_6(bool do_add) throw(libtest::test_exception) {

	//
	//	Single operand A + B and C + D, symmetry
	//

	static const char *testname = "btod_sum_test::test_6()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<4> i1, i2;
	i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
	dimensions<4> dims_iiaa(index_range<4>(i1, i2));
	i2[0] = 12; i2[1] = 6; i2[2] = 6; i2[3] = 6;
	block_index_space<4> bis_iiaa(dims_iiaa);
	mask<4> m1, m2;
	m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
	bis_iiaa.split(m1, 3);
	bis_iiaa.split(m1, 7);
	bis_iiaa.split(m1, 10);
	bis_iiaa.split(m2, 2);
	bis_iiaa.split(m2, 3);
	bis_iiaa.split(m2, 5);

	block_tensor<4, double, allocator_t> bta1(bis_iiaa), bta2(bis_iiaa);
	block_tensor<4, double, allocator_t> btb1(bis_iiaa), btb2(bis_iiaa);
	block_tensor<4, double, allocator_t> btc(bis_iiaa), btc_ref(bis_iiaa);

	{
	block_tensor_ctrl<4, double> ctrl_a1(bta1), ctrl_a2(bta2);
	block_tensor_ctrl<4, double> ctrl_b1(btb1), ctrl_b2(btb2);
	block_tensor_ctrl<4, double> ctrl_c(btc), ctrl_c_ref(btc_ref);
	permutation<4> p1023, p0132;
	p1023.permute(0, 1);
	p0132.permute(2, 3);
	se_perm<4, double> sp1023(p1023, true), sp0132(p0132, false);
	ctrl_a1.req_symmetry().insert(sp1023);
	ctrl_a1.req_symmetry().insert(sp0132);
	ctrl_b1.req_symmetry().insert(sp1023);
	ctrl_b1.req_symmetry().insert(sp0132);
	ctrl_a2.req_symmetry().insert(sp1023);
	ctrl_b2.req_symmetry().insert(sp1023);
	ctrl_c.req_symmetry().insert(sp1023);
	ctrl_c.req_symmetry().insert(sp0132);
	ctrl_c_ref.req_symmetry().insert(sp1023);
	ctrl_c_ref.req_symmetry().insert(sp0132);
	}

	//	Load random data for input

	btod_random<4>().perform(bta1);
	btod_random<4>().perform(btb1);
	btod_random<4>().perform(bta2);
	btod_random<4>().perform(btb2);
	bta1.set_immutable();
	btb1.set_immutable();
	bta2.set_immutable();
	btb2.set_immutable();

	// Prepare reference

	if (do_add) {
		btod_random<4>().perform(btc);
		btod_copy<4>(btc).perform(btc_ref);
	}

	//	Run contraction and compute the reference

	btod_add<4> op1(bta1), op2(bta2);
	op1.add_op(btb1);
	op2.add_op(btb2);
	if (do_add) op1.perform(btc_ref, 1.0);
	else op1.perform(btc_ref);
	op2.perform(btc_ref, 1.0);

	btod_sum<4> sum(op1);
	sum.add_op(op2);
	if (do_add) sum.perform(btc, 1.0);
	else sum.perform(btc);

	compare_ref<4>::compare(testname, btc, btc_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

