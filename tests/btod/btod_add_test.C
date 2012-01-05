#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include "btod_add_test.h"
#include "../compare_ref.h"

namespace libtensor {

void btod_add_test::perform() throw(libtest::test_exception) {

	test_1(1.0, 1.0);
	test_1(1.0, 0.5);
	test_1(0.5, 1.0);
	test_1(2.5, 1.5);

	test_2(1.0, 1.0, 1.0);
	test_2(1.0, 0.5, 1.0);
	test_2(0.5, 1.0, 1.0);
	test_2(2.5, 1.5, 1.0);
	test_2(1.0, 1.0, -1.0);
	test_2(1.0, 0.5, -1.0);
	test_2(0.5, 1.0, -1.0);
	test_2(2.5, 1.5, -1.0);

	test_3(2.0, 1.0);
	test_3(1.0, 1.0);
	test_4(2.0, 1.0, 1.0, 1.0);
	test_4(1.0, 1.0, 1.0, 1.0);

	test_5();
	test_6();
	test_7();
	test_8();
	test_9();

	test_exc();
}

void btod_add_test::test_1(double ca1, double ca2)
	throw(libtest::test_exception) {

	//
	//	Arg 1: One non-zero block
	//	Arg 2: One non-zero block, permuted
	//	Simple run (replacement of data in B)
	//

	std::ostringstream tnss;
	tnss << "btod_add_test::test_1(" << ca1 << ", " << ca2 << ")";

	typedef std_allocator<double> allocator_t;

	cpu_pool cpus(1);

	try {

	index<2> i0, i1, i2;
	i2[0] = 10; i2[1] = 20;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 20; i2[1] = 10;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> bis_ia(dims_ia), bis_ai(dims_ai);

	permutation<2> perma1, perma2;
	perma2.permute(0, 1);

	block_tensor<2, double, allocator_t> bta1(bis_ia), bta2(bis_ai),
		btb(bis_ia);

	//	Fill in random input

	btod_random<2>().perform(bta1);
	btod_random<2>().perform(bta2);

	//	Prepare reference data

	dense_tensor<2, double, allocator_t> ta1(dims_ia), ta2(dims_ai),
		tb(dims_ia), tb_ref(dims_ia);
	tod_btconv<2>(bta1).perform(ta1);
	tod_btconv<2>(bta2).perform(ta2);
	tod_add<2> op_ref(ta1, perma1, ca1);
	op_ref.add_op(ta2, perma2, ca2);
	op_ref.perform(cpus, true, 1.0, tb_ref);

	//	Run the addition operation

	btod_add<2> op(bta1, perma1, ca1);
	op.add_op(bta2, perma2, ca2);
	op.perform(btb);

	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference

	compare_ref<2>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


void btod_add_test::test_2(double ca1, double ca2, double cs)
	throw(libtest::test_exception) {

	//
	//	Arg 1: One non-zero block
	//	Arg 2: One non-zero block, permuted
	//	Additive run
	//

	std::ostringstream tnss;
	tnss << "btod_add_test::test_2(" << ca1 << ", " << ca2
		<< ", " << cs << ")";

	typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

	try {

	index<2> i0, i1, i2;
	i2[0] = 10; i2[1] = 20;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 20; i2[1] = 10;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> bis_ia(dims_ia), bis_ai(dims_ai);

	permutation<2> perma1, perma2;
	perma2.permute(0, 1);

	block_tensor<2, double, allocator_t> bta1(bis_ia), bta2(bis_ai),
		btb(bis_ia);

	//	Fill in random input

	btod_random<2>().perform(bta1);
	btod_random<2>().perform(bta2);
	btod_random<2>().perform(btb);

	//	Prepare reference data

	dense_tensor<2, double, allocator_t> ta1(dims_ia), ta2(dims_ai),
		tb(dims_ia), tb_ref(dims_ia);
	tod_btconv<2>(bta1).perform(ta1);
	tod_btconv<2>(bta2).perform(ta2);
	tod_btconv<2>(btb).perform(tb_ref);
	tod_add<2> op_ref(ta1, perma1, ca1);
	op_ref.add_op(ta2, perma2, ca2);
	op_ref.perform(cpus, false, cs, tb_ref);

	//	Run the addition operation

	btod_add<2> op(bta1, perma1, ca1);
	op.add_op(bta2, perma2, ca2);
	op.perform(btb, cs);

	tod_btconv<2>(btb).perform(tb);

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

	typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> splmsk; splmsk[0] = true; splmsk[1] = true;
	bis.split(splmsk, 3);

	block_tensor<2, double, allocator_t> bta1(bis), bta2(bis), btb(bis);
	block_tensor_ctrl<2, double> ctrl_bta1(bta1), ctrl_bta2(bta2),
		ctrl_btb(btb);

	se_perm<2, double> cycle(permutation<2>().permute(0, 1), true);
	ctrl_bta1.req_symmetry().insert(cycle);
	ctrl_bta2.req_symmetry().insert(cycle);
	ctrl_btb.req_symmetry().insert(cycle);

	index<2> i_00, i_01, i_11;
	i_01[0] = 0; i_01[1] = 1;
	i_11[0] = 1; i_11[1] = 1;

	//	Fill in random input

	btod_random<2>().perform(bta1, i_01);
	btod_random<2>().perform(bta2, i_00);
	btod_random<2>().perform(bta2, i_11);

	//	 Prepare the reference

	dense_tensor<2, double, allocator_t> ta1(dims), ta2(dims), tb(dims),
		tb_ref(dims);

	tod_btconv<2>(bta1).perform(ta1);
	tod_btconv<2>(bta2).perform(ta2);

	tod_add<2> op_ref(ta1, ca1);
	op_ref.add_op(ta2, ca2);
	op_ref.perform(cpus, true, 1.0, tb_ref);

	//	Run the addition operation

	btod_add<2> op(bta1, ca1);
	op.add_op(bta2, ca2);
	op.perform(btb);

	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference

	compare_ref<2>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


void btod_add_test::test_4(double ca1, double ca2, double ca3, double ca4)
	throw(libtest::test_exception) {

	//
	//	Arg 1: One non-zero off-diagonal block [0,0,0,1]
	//	Arg 2: Non-zero diagonal blocks [0,0,0,0] and [1,1,1,1]
	//	Arg 3: One non-zero off-diagonal block [0,0,1,1]
	//	Arg 4: One non-zero off-diagonal block, permuted [0,1,1,1]
	//	All arguments have the same permutational symmetry
	//

	std::ostringstream tnss;
	tnss << "btod_add_test::test_4(" << ca1 << ", " << ca2
		<< ", " << ca3 << ", " << ca4 << ")";

	typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> splmsk;
	splmsk[0] = true; splmsk[1] = true; splmsk[2] = true; splmsk[3] = true;
	bis.split(splmsk, 3);

	block_tensor<4, double, allocator_t> bta1(bis), bta2(bis), bta3(bis),
		bta4(bis), btb(bis);
	block_tensor_ctrl<4, double> ctrl_bta1(bta1), ctrl_bta2(bta2),
		ctrl_bta3(bta3), ctrl_bta4(bta4), ctrl_btb(btb);

	se_perm<4, double> cycle1(permutation<4>().permute(0, 1).permute(1, 2).
		permute(2, 3), true);
	se_perm<4, double> cycle2(permutation<4>().permute(0, 1), true);
	ctrl_bta1.req_symmetry().insert(cycle1);
	ctrl_bta1.req_symmetry().insert(cycle2);
	ctrl_bta2.req_symmetry().insert(cycle1);
	ctrl_bta2.req_symmetry().insert(cycle2);
	ctrl_bta3.req_symmetry().insert(cycle1);
	ctrl_bta3.req_symmetry().insert(cycle2);
	ctrl_bta4.req_symmetry().insert(cycle1);
	ctrl_bta4.req_symmetry().insert(cycle2);
	ctrl_btb.req_symmetry().insert(cycle1);
	ctrl_btb.req_symmetry().insert(cycle2);

	index<4> i_0000, i_0001, i_0011, i_0111, i_1111;
	i_0001[0] = 0; i_0001[1] = 0; i_0001[2] = 0; i_0001[3] = 1;
	i_0011[0] = 0; i_0011[1] = 0; i_0011[2] = 1; i_0011[3] = 1;
	i_0111[0] = 0; i_0111[1] = 1; i_0111[2] = 1; i_0111[3] = 1;
	i_1111[0] = 1; i_1111[1] = 1; i_1111[2] = 1; i_1111[3] = 1;

	//	Fill in random input

	btod_random<4> rand;
	rand.perform(bta1, i_0001);
	rand.perform(bta2, i_0000);
	rand.perform(bta2, i_1111);
	rand.perform(bta3, i_0011);
	rand.perform(bta4, i_0111);

	bta1.set_immutable();
	bta2.set_immutable();
	bta3.set_immutable();
	bta4.set_immutable();

	//	 Prepare the reference

	dense_tensor<4, double, allocator_t> ta1(dims), ta2(dims), ta3(dims),
		ta4(dims), tb(dims), tb_ref(dims);
	tod_btconv<4>(bta1).perform(ta1);
	tod_btconv<4>(bta2).perform(ta2);
	tod_btconv<4>(bta3).perform(ta3);
	tod_btconv<4>(bta4).perform(ta4);

	permutation<4> perm4; perm4.permute(1, 2).permute(2, 3);
	tod_add<4> op_ref(ta1, ca1);
	op_ref.add_op(ta2, ca2);
	op_ref.add_op(ta3, ca3);
	op_ref.add_op(ta4, perm4, ca4);
	op_ref.perform(cpus, true, 1.0, tb_ref);

	//	Run the addition operation

	btod_add<4> op(bta1, ca1);
	op.add_op(bta2, ca2);
	op.add_op(bta3, ca3);
	op.add_op(bta4, perm4, ca4);
	op.perform(btb);

	tod_btconv<4>(btb).perform(tb);

	//	Compare against the reference

	compare_ref<4>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


void btod_add_test::test_5() throw(libtest::test_exception) {

	//
	//	Tests addition to zero vs. overwrite (single arguments)
	//

	static const char *testname = "btod_add_test::test_5()";

	typedef std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	block_tensor_t bt1(bis), bt3(bis), bt3_ref(bis);
	btod_random<2>().perform(bt1);
	bt1.set_immutable();

	btod_add<2> add(bt1);

	add.perform(bt3, 1.0);
	add.perform(bt3_ref);

	compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_add_test::test_6() throw(libtest::test_exception) {

	//
	//	Tests addition to zero vs. overwrite (multiple arguments)
	//

	static const char *testname = "btod_add_test::test_6()";

	typedef std_allocator<double> allocator_t;
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

	add.perform(bt3, 1.0);
	add.perform(bt3_ref);

	compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\brief Tests a particular block %index space that is causing a problem
 **/
void btod_add_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "btod_add_test::test_7()";

	typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

	try {

	size_t ni = 13, na = 7;
	index<4> i1, i2;
	i2[0] = na - 1; i2[1] = na - 1; i2[2] = ni - 1; i2[3] = na - 1;
	dimensions<4> dims_caib(index_range<4>(i1, i2));
	block_index_space<4> bis_caib(dims_caib);
	i2[0] = ni - 1; i2[1] = na - 1; i2[2] = na - 1; i2[3] = na - 1;
	dimensions<4> dims_iabc(index_range<4>(i1, i2));
	block_index_space<4> bis_iabc(dims_iabc);

	mask<4> m1, m2;
	m1[0] = true; m1[1] = true; m1[3] = true;
	bis_caib.split(m1, 2);
	bis_caib.split(m1, 3);
	bis_caib.split(m1, 5);
	m2[1] = true; m2[2] = true; m2[3] = true;
	bis_iabc.split(m2, 2);
	bis_iabc.split(m2, 3);
	bis_iabc.split(m2, 5);

	block_tensor<4, double, allocator_t> bt1(bis_caib), bt2(bis_caib),
		bt3(bis_iabc);
	dense_tensor<4, double, allocator_t> t1(dims_caib), t2(dims_caib),
		t3(dims_iabc), t3_ref(dims_iabc);

	btod_random<4>().perform(bt1);
	btod_random<4>().perform(bt2);
	bt1.set_immutable();
	bt2.set_immutable();

	tod_btconv<4>(bt1).perform(t1);
	tod_btconv<4>(bt2).perform(t2);

	permutation<4> p_caib, p_baic;
	p_caib.permute(0, 2).permute(2, 3); // caib -> iabc
	p_baic.permute(0, 2); // baic -> iabc

	btod_add<4> add(bt1, p_caib, 1.0);
	add.add_op(bt2, p_baic, -1.0);
	add.perform(bt3);
	tod_btconv<4>(bt3).perform(t3);

	tod_add<4> addt(t1, p_caib, 1.0);
	addt.add_op(t2, p_baic, -1.0);
	addt.perform(cpus, true, 1.0, t3_ref);

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\brief Tests \f$ B_{iajb} = B_{iajb} + A_{ijab} qquad
		A \in S^{-}_2 \times S^{-}_2 \f$
 **/
void btod_add_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "btod_add_test::test_8()";

	typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

	try {

	size_t ni = 10, na = 4;
	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = na - 1; i2[2] = ni - 1; i2[3] = na - 1;
	dimensions<4> dims_iajb(index_range<4>(i1, i2));
	block_index_space<4> bis_iajb(dims_iajb);
	i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = na - 1; i2[3] = na - 1;
	dimensions<4> dims_ijab(index_range<4>(i1, i2));
	block_index_space<4> bis_ijab(dims_ijab);

	mask<4> m1, m2, m3, m4;
	m1[0] = true; m2[1] = true; m1[2] = true; m2[3] = true;
	m3[0] = true; m3[1] = true; m4[2] = true; m4[3] = true;
	bis_iajb.split(m1, 5);
	bis_iajb.split(m2, 2);
	bis_ijab.split(m3, 5);
	bis_ijab.split(m4, 2);

	block_tensor<4, double, allocator_t> bta(bis_ijab), btb(bis_iajb);
	dense_tensor<4, double, allocator_t> ta(dims_ijab), tb(dims_iajb),
		tb_ref(dims_iajb);
	symmetry<4, double> syma_ref(bis_iajb);

	{
		block_tensor_ctrl<4, double> ca(bta);
		ca.req_symmetry().insert(se_perm<4, double>(permutation<4>().
			permute(0, 1), false));
		ca.req_symmetry().insert(se_perm<4, double>(permutation<4>().
			permute(2, 3), false));
	}

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();

	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);
	tod_copy<4>(ta, permutation<4>().permute(1, 2), 1.0).
		perform(cpus, false, 1.0, tb_ref);
	syma_ref.insert(se_perm<4, double>(permutation<4>().
		permute(0, 2), false));
	syma_ref.insert(se_perm<4, double>(permutation<4>().
		permute(1, 3), false));

	btod_add<4> add(bta, permutation<4>().permute(1, 2), 1.0);

	compare_ref<4>::compare(testname, add.get_symmetry(), syma_ref);

	add.perform(btb, 1.0);
	tod_btconv<4>(btb).perform(tb);

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\brief Tests \f$ B_{jkab} = B_{jkab} + A_{kajb} qquad
		A \in S^{-}_2 \times S^{-}_2 \f$
 **/
void btod_add_test::test_9() throw(libtest::test_exception) {

	static const char *testname = "btod_add_test::test_9()";

	typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

	try {

	size_t ni = 10, na = 4;
	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = na - 1; i2[2] = ni - 1; i2[3] = na - 1;
	dimensions<4> dims_iajb(index_range<4>(i1, i2));
	block_index_space<4> bis_iajb(dims_iajb);
	i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = na - 1; i2[3] = na - 1;
	dimensions<4> dims_ijab(index_range<4>(i1, i2));
	block_index_space<4> bis_ijab(dims_ijab);

	mask<4> m1, m2, m3, m4;
	m1[0] = true; m2[1] = true; m1[2] = true; m2[3] = true;
	m3[0] = true; m3[1] = true; m4[2] = true; m4[3] = true;
	bis_iajb.split(m1, 5);
	bis_iajb.split(m2, 2);
	bis_ijab.split(m3, 5);
	bis_ijab.split(m4, 2);

	block_tensor<4, double, allocator_t> bta(bis_iajb), btb(bis_ijab);
	dense_tensor<4, double, allocator_t> ta(dims_iajb), tb(dims_ijab),
		tb_ref(dims_ijab);
	symmetry<4, double> syma_ref(bis_ijab);

	{
		block_tensor_ctrl<4, double> ca(bta);
		ca.req_symmetry().insert(se_perm<4, double>(permutation<4>().
			permute(0, 2).permute(1, 3), true));
	}

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();

	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);
	tod_copy<4>(ta, permutation<4>().permute(2, 1).permute(1, 0), 1.0).
		perform(cpus, false, 1.0, tb_ref);
	syma_ref.insert(se_perm<4, double>(permutation<4>().
		permute(0, 1).permute(2, 3), true));

	btod_add<4> add(bta, permutation<4>().permute(2, 1).permute(1, 0), 1.0);

	compare_ref<4>::compare(testname, add.get_symmetry(), syma_ref);

	add.perform(btb, 1.0);
	tod_btconv<4>(btb).perform(tb);

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
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

