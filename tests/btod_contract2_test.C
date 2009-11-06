#include <sstream>
#include <libvmm.h>
#include <libtensor.h>
#include "btod_contract2_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_contract2_test::perform() throw(libtest::test_exception) {

	test_bis_1();
	test_bis_2();
	test_sym_1();
	test_sym_2();
	test_contr_1();
	test_contr_2();
	test_contr_3();
	test_contr_4();
	test_contr_5();
//	test_contr_6();
	test_contr_7();
	test_contr_8();
	test_contr_9();
	test_contr_10();
	test_contr_11();
	test_contr_12();
	test_contr_13();
}


void btod_contract2_test::test_bis_1() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ijpq b_klpq
	//

	static const char *testname = "btod_contract2_test::test_bis_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bisa(dims), bis_ref(dims);
	mask<4> msk, msk1, msk2;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk, 3);
	bisa.split(msk, 5);
	bis_ref.split(msk1, 3);
	bis_ref.split(msk1, 5);
	bis_ref.split(msk2, 3);
	bis_ref.split(msk2, 5);

	block_index_space<4> bisb(bisa);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);
	contraction2<2, 2, 2> contr;
	contr.contract(0, 2);
	contr.contract(1, 3);

	btod_contract2<2, 2, 2> op(contr, bta, btb);

	if(!op.get_bis().equals(bis_ref)) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected output block index space.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_bis_2() throw(libtest::test_exception) {

	//
	//	c_ijk = a_ipqr b_jpqrk
	//

	static const char *testname = "btod_contract2_test::test_bis_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<3> i3_1, i3_2;
	i3_2[0] = 10; i3_2[1] = 10; i3_2[2] = 8;
	dimensions<3> dims3(index_range<3>(i3_1, i3_2));
	index<4> i4_1, i4_2;
	i4_2[0] = 10; i4_2[1] = 10; i4_2[2] = 10; i4_2[3] = 10;
	dimensions<4> dims4(index_range<4>(i4_1, i4_2));
	index<5> i5_1, i5_2;
	i5_2[0] = 10; i5_2[1] = 10; i5_2[2] = 10; i5_2[3] = 10; i5_2[4] = 8;
	dimensions<5> dims5(index_range<5>(i5_1, i5_2));

	block_index_space<4> bisa(dims4);
	block_index_space<5> bisb(dims5);
	block_index_space<3> bis_ref(dims3);

	mask<3> msk3_1, msk3_2, msk3_3;
	msk3_1[0] = true; msk3_2[1] = true; msk3_3[2] = true;
	mask<4> msk4;
	msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
	mask<5> msk5_1, msk5_2;
	msk5_1[0] = true; msk5_1[1] = true; msk5_1[2] = true; msk5_1[3] = true;
	msk5_2[4] = true;

	bisa.split(msk4, 3);
	bisa.split(msk4, 5);
	bisb.split(msk5_1, 3);
	bisb.split(msk5_1, 5);
	bisb.split(msk5_2, 4);
	bis_ref.split(msk3_1, 3);
	bis_ref.split(msk3_1, 5);
	bis_ref.split(msk3_2, 3);
	bis_ref.split(msk3_2, 5);
	bis_ref.split(msk3_3, 4);

	block_tensor<4, double, allocator_t> bta(bisa);
	block_tensor<5, double, allocator_t> btb(bisb);
	contraction2<1, 2, 3> contr;
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);

	btod_contract2<1, 2, 3> op(contr, bta, btb);

	if(!op.get_bis().equals(bis_ref)) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected output block index space.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_sym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_contract2_test::test_sym_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bisa(dims), bis_ref(dims);
	mask<4> msk, msk1, msk2;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk, 3);
	bisa.split(msk, 5);
	bis_ref.split(msk1, 3);
	bis_ref.split(msk1, 5);
	bis_ref.split(msk2, 3);
	bis_ref.split(msk2, 5);

	block_index_space<4> bisb(bisa);
	dimensions<4> bidimsa(bisa.get_block_index_dims()),
		bidimsb(bisb.get_block_index_dims()),
		bidimsc(bis_ref.get_block_index_dims());

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	mask<4> cyclemsk4, cyclemsk2_1, cyclemsk2_2;
	cyclemsk4[0] = true; cyclemsk4[1] = true;
	cyclemsk4[2] = true; cyclemsk4[3] = true;
	cyclemsk2_1[0] = true; cyclemsk2_1[1] = true;
	cyclemsk2_2[2] = true; cyclemsk2_2[3] = true;

	symel_cycleperm<4, double> cycle4a(4, cyclemsk4),
		cycle2a(2, cyclemsk2_1);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_sym_add_element(cycle4a);
	ctrla.req_sym_add_element(cycle2a);
	ctrlb.req_sym_add_element(cycle4a);
	ctrlb.req_sym_add_element(cycle2a);

	symmetry<4, double> sym_ref(bis_ref);
	symel_cycleperm<4, double> cycle2c_1(2, cyclemsk2_1),
		cycle2c_2(2, cyclemsk2_2);
	sym_ref.add_element(cycle2c_1);
	sym_ref.add_element(cycle2c_2);

	contraction2<2, 2, 2> contr;
	contr.contract(0, 2);
	contr.contract(1, 3);

	btod_contract2<2, 2, 2> op(contr, bta, btb);

	if(!op.get_symmetry().equals(sym_ref)) {
		fail_test(testname, __FILE__, __LINE__,
			"Symmetry does not match reference.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_sym_2() throw(libtest::test_exception) {

	//
	//	c_ijk = a_ipqr b_jpqrk
	//	Permutational symmetry in ijpqr
	//

	static const char *testname = "btod_contract2_test::test_sym_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<3> i3_1, i3_2;
	i3_2[0] = 10; i3_2[1] = 10; i3_2[2] = 8;
	dimensions<3> dims3(index_range<3>(i3_1, i3_2));
	index<4> i4_1, i4_2;
	i4_2[0] = 10; i4_2[1] = 10; i4_2[2] = 10; i4_2[3] = 10;
	dimensions<4> dims4(index_range<4>(i4_1, i4_2));
	index<5> i5_1, i5_2;
	i5_2[0] = 10; i5_2[1] = 10; i5_2[2] = 10; i5_2[3] = 10; i5_2[4] = 8;
	dimensions<5> dims5(index_range<5>(i5_1, i5_2));

	block_index_space<4> bisa(dims4);
	block_index_space<5> bisb(dims5);
	block_index_space<3> bis_ref(dims3);

	mask<3> msk3_1, msk3_2, msk3_3;
	msk3_1[0] = true; msk3_2[1] = true; msk3_3[2] = true;
	mask<4> msk4;
	msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
	mask<5> msk5_1, msk5_2;
	msk5_1[0] = true; msk5_1[1] = true; msk5_1[2] = true; msk5_1[3] = true;
	msk5_2[4] = true;

	bisa.split(msk4, 3);
	bisa.split(msk4, 5);
	bisb.split(msk5_1, 3);
	bisb.split(msk5_1, 5);
	bisb.split(msk5_2, 4);
	bis_ref.split(msk3_1, 3);
	bis_ref.split(msk3_1, 5);
	bis_ref.split(msk3_2, 3);
	bis_ref.split(msk3_2, 5);
	bis_ref.split(msk3_3, 4);

	block_tensor<4, double, allocator_t> bta(bisa);
	block_tensor<5, double, allocator_t> btb(bisb);

	mask<4> cyclemsk4_4;
	cyclemsk4_4[0] = true; cyclemsk4_4[1] = true;
	cyclemsk4_4[2] = true; cyclemsk4_4[3] = true;
	mask<5> cyclemsk5_4;
	cyclemsk5_4[0] = true; cyclemsk5_4[1] = true;
	cyclemsk5_4[2] = true; cyclemsk5_4[3] = true;

	symmetry<3, double> sym_ref(bis_ref);
	symel_cycleperm<4, double> cycle4a_1(4, cyclemsk4_4),
		cycle4a_2(2, cyclemsk4_4);
	symel_cycleperm<5, double> cycle4b_1(4, cyclemsk5_4),
		cycle4b_2(2, cyclemsk5_4);

	block_tensor_ctrl<4, double> ctrla(bta);
	block_tensor_ctrl<5, double> ctrlb(btb);
	ctrla.req_sym_add_element(cycle4a_1);
	ctrla.req_sym_add_element(cycle4a_2);
	ctrlb.req_sym_add_element(cycle4b_1);
	ctrlb.req_sym_add_element(cycle4b_2);

	contraction2<1, 2, 3> contr;
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);

	btod_contract2<1, 2, 3> op(contr, bta, btb);

	if(!op.get_symmetry().equals(sym_ref)) {
		fail_test(testname, __FILE__, __LINE__,
			"Symmetry does not match reference.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_1() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ijpq b_klpq
	//	All dimensions are identical, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bisa(dims), bisc(dims);
	mask<4> msk, msk1, msk2;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk, 3);
	bisa.split(msk, 5);
	bisc.split(msk1, 3);
	bisc.split(msk1, 5);
	bisc.split(msk2, 3);
	bisc.split(msk2, 5);

	block_index_space<4> bisb(bisa);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Run contraction

	contraction2<2, 2, 2> contr;
	contr.contract(2, 2);
	contr.contract(3, 3);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc);

	//	Convert block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);
	tod_btconv<4> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_2() throw(libtest::test_exception) {

	//
	//	c_ikjl = a_ijpq b_klqp
	//	All dimensions are identical, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bisa(dims), bisc(dims);
	mask<4> msk, msk1, msk2, msk1c, msk2c;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	msk1c[0] = true; msk1c[2] = true;
	msk2c[1] = true; msk2c[3] = true;

	bisa.split(msk, 3);
	bisa.split(msk, 5);
	bisc.split(msk1c, 3);
	bisc.split(msk1c, 5);
	bisc.split(msk2c, 3);
	bisc.split(msk2c, 5);

	block_index_space<4> bisb(bisa);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Run contraction

	permutation<4> permc; permc.permute(1, 2);
	contraction2<2, 2, 2> contr(permc);
	contr.contract(2, 3);
	contr.contract(3, 2);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc);

	//	Convert block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);
	tod_btconv<4> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_3() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ijpq b_pqkl
	//	Dimensions [ij]=10, [kl]=12, [pq]=6, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsb(index_range<4>(i1, i2));
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsc(index_range<4>(i1, i2));
	block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 4);
	bisb.split(msk2, 6);

	bisc.split(msk1, 3);
	bisc.split(msk1, 5);
	bisc.split(msk2, 6);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Run contraction

	contraction2<2, 2, 2> contr;
	contr.contract(2, 0);
	contr.contract(3, 1);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc);

	//	Convert block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
		tc_ref(dimsc);
	tod_btconv<4> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_4() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ijpq b_pqkl
	//	Dimensions [ij]=10, [kl]=12, [pq]=6, permutational symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsb(index_range<4>(i1, i2));
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsc(index_range<4>(i1, i2));
	block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 4);
	bisb.split(msk2, 6);

	bisc.split(msk1, 3);
	bisc.split(msk1, 5);
	bisc.split(msk2, 6);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

	//	Set up symmetry

	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_sym_add_element(cycle1);
	ctrla.req_sym_add_element(cycle2);
	ctrlb.req_sym_add_element(cycle1);
	ctrlb.req_sym_add_element(cycle2);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Run contraction

	contraction2<2, 2, 2> contr;
	contr.contract(2, 0);
	contr.contract(3, 1);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc);

	//	Convert block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
		tc_ref(dimsc);
	tod_btconv<4> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_5() throw(libtest::test_exception) {

	//
	//	c_ijkl = c_ijkl + a_ijpq b_pqkl
	//	Dimensions [ij]=10, [kl]=12, [pq]=6, permutational symmetry
	//	Sym(C) = Sym(A*B)
	//

	static const char *testname = "btod_contract2_test::test_contr_5()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsb(index_range<4>(i1, i2));
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsc(index_range<4>(i1, i2));
	block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 4);
	bisb.split(msk2, 6);

	bisc.split(msk1, 3);
	bisc.split(msk1, 5);
	bisc.split(msk2, 6);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

	//	Set up symmetry

	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlc(btc);
	ctrla.req_sym_add_element(cycle1);
	ctrla.req_sym_add_element(cycle2);
	ctrlb.req_sym_add_element(cycle1);
	ctrlb.req_sym_add_element(cycle2);
	ctrlc.req_sym_add_element(cycle1);
	ctrlc.req_sym_add_element(cycle2);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	rand.perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert input block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
		tc_ref(dimsc);
	tod_btconv<4> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc_ref(btc);
	convc_ref.perform(tc_ref);

	//	Run contraction

	contraction2<2, 2, 2> contr;
	contr.contract(2, 0);
	contr.contract(3, 1);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc, 2.0);

	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref, 2.0);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_6() throw(libtest::test_exception) {

	//
	//	c_ijkl = c_ijkl + a_ijpq b_pqkl
	//	Dimensions [ij]=10, [kl]=12, [pq]=6, permutational symmetry
	//	Sym(C) > Sym(A*B)
	//

	static const char *testname = "btod_contract2_test::test_contr_6()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsb(index_range<4>(i1, i2));
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsc(index_range<4>(i1, i2));
	block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 4);
	bisb.split(msk2, 6);

	bisc.split(msk1, 3);
	bisc.split(msk1, 5);
	bisc.split(msk2, 6);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

	//	Set up symmetry

	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlc(btc);
	ctrla.req_sym_add_element(cycle2);
	ctrlb.req_sym_add_element(cycle1);
	ctrlb.req_sym_add_element(cycle2);
	ctrlc.req_sym_add_element(cycle1);
	ctrlc.req_sym_add_element(cycle2);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	rand.perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert input block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
		tc_ref(dimsc);
	tod_btconv<4> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc_ref(btc);
	convc_ref.perform(tc_ref);

	//	Run contraction

	contraction2<2, 2, 2> contr;
	contr.contract(2, 0);
	contr.contract(3, 1);

	btod_contract2<2, 2, 2> op(contr, bta, btb);
	op.perform(btc, 2.0);

	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref, 2.0);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_7() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_pi b_jklp
	//	Dimensions [ijkl]=10, [p]=6, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_7()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i21, i22;
	i22[0] = 5; i22[1] = 9;
	dimensions<2> dimsa(index_range<2>(i21, i22));
	index<4> i41, i42;
	i42[0] = 9; i42[1] = 9; i42[2] = 9; i42[3] = 5;
	dimensions<4> dimsb(index_range<4>(i41, i42));
	i42[0] = 9; i42[1] = 9; i42[2] = 9; i42[3] = 9;
	dimensions<4> dimsc(index_range<4>(i41, i42));
	block_index_space<2> bisa(dimsa);
	block_index_space<4> bisb(dimsb), bisc(dimsc);

	mask<2> mska;
	mask<4> mskb, mskc;
	mska[1] = true;
	mskb[0] = true; mskb[1] = true; mskb[2] = true;
	mskc[0] = true; mskc[1] = true; mskc[2] = true; mskc[3] = true;

	bisa.split(mska, 3);
	bisb.split(mskb, 3);
	bisc.split(mskc, 3);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

	//	Load random data for input

	btod_random<2>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Run contraction

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);

	btod_contract2<1, 3, 1> op(contr, bta, btb);
	op.perform(btc);

	//	Convert block tensors to regular tensors

	tensor<2, double, allocator_t> ta(dimsa);
	tensor<4, double, allocator_t> tb(dimsb), tc(dimsc), tc_ref(dimsc);
	tod_btconv<2> conva(bta);
	conva.perform(ta);
	tod_btconv<4> convb(btb);
	convb.perform(tb);
	tod_btconv<4> convc(btc);
	convc.perform(tc);

	//	Compute reference tensor

	tod_contract2<1, 3, 1> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_8() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_pi b_jklp
	//	Dimensions [ijkl]=10, [p]=6, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_8()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i21, i22;
	i22[0] = 9; i22[1] = 19;
	dimensions<2> dimsa(index_range<2>(i21, i22));
	index<4> i41, i42;
	i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 9;
	dimensions<4> dimsb(index_range<4>(i41, i42));
	i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 19;
	dimensions<4> dimsc(index_range<4>(i41, i42));
	block_index_space<2> bisa(dimsa);
	block_index_space<4> bisb(dimsb), bisc(dimsc);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

	//	Load random data for input

	btod_random<2>().perform(bta);
	btod_random<4>().perform(btb);
	btod_random<4>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert block tensors to regular tensors

	tensor<2, double, allocator_t> ta(dimsa);
	tensor<4, double, allocator_t> tb(dimsb), tc(dimsc), tc_ref(dimsc);
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb);
	tod_btconv<4>(btc).perform(tc_ref);

	//	Run contraction

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);

	btod_contract2<1, 3, 1> op(contr, bta, btb);
	op.perform(btc, 1.0);

	tod_btconv<4>(btc).perform(tc);

	//	Compute reference tensor

	tod_contract2<1, 3, 1> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref, 1.0);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_9() throw(libtest::test_exception) {

	//
	//	c_ijkl = - a_pi b_jklp
	//	Dimensions [ijkl]=10, [p]=6, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_9()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i21, i22;
	i22[0] = 9; i22[1] = 19;
	dimensions<2> dimsa(index_range<2>(i21, i22));
	index<4> i41, i42;
	i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 9;
	dimensions<4> dimsb(index_range<4>(i41, i42));
	i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 19;
	dimensions<4> dimsc(index_range<4>(i41, i42));
	block_index_space<2> bisa(dimsa);
	block_index_space<4> bisb(dimsb), bisc(dimsc);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

	//	Load random data for input

	btod_random<2>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert block tensors to regular tensors

	tensor<2, double, allocator_t> ta(dimsa);
	tensor<4, double, allocator_t> tb(dimsb), tc(dimsc), tc_ref(dimsc);
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb);
	tod_btconv<4>(btc).perform(tc_ref);

	//	Run contraction

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);

	btod_contract2<1, 3, 1> op(contr, bta, btb);
	op.perform(btc, -1.0);

	tod_btconv<4>(btc).perform(tc);

	//	Compute reference tensor

	tod_contract2<1, 3, 1> op_ref(contr, ta, tb);
	op_ref.perform(tc_ref, -1.0);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_10() throw(libtest::test_exception) {

	//
	//	c_ijkl = - a_pi b_jklp
	//	Dimensions [ijkl]=10, [p]=6, no symmetry
	//	Copy with -1.0 vs. a coefficient in btod_contract2
	//

	static const char *testname = "btod_contract2_test::test_contr_10()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i21, i22;
	i22[0] = 9; i22[1] = 19;
	dimensions<2> dimsa(index_range<2>(i21, i22));
	index<4> i41, i42;
	i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 9;
	dimensions<4> dimsb(index_range<4>(i41, i42));
	i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 19;
	dimensions<4> dimsc(index_range<4>(i41, i42));
	block_index_space<2> bisa(dimsa);
	block_index_space<4> bisb(dimsb), bisc(dimsc);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<4, double, allocator_t> btb(bisb);
	block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc),
		btc_ref_tmp(bisc);

	//	Load random data for input

	btod_random<2>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert block tensors to regular tensors

	//	Run contraction and compute the reference

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);

	btod_contract2<1, 3, 1>(contr, bta, btb).perform(btc, -1.0);
	btod_contract2<1, 3, 1>(contr, bta, btb).perform(btc_ref_tmp);
	btod_copy<4>(btc_ref_tmp, -1.0).perform(btc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, btc, btc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_11() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ij b_kl
	//	Dimensions [ij] = 10, [kl]=20, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_11()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i21, i22;
	i22[0] = 9; i22[1] = 9;
	dimensions<2> dimsa(index_range<2>(i21, i22));
	i22[0] = 19; i22[1] = 19;
	dimensions<2> dimsb(index_range<2>(i21, i22));
	index<4> i41, i42;
	i42[0] = 9; i42[1] = 9; i42[2] = 19; i42[3] = 19;
	dimensions<4> dimsc(index_range<4>(i41, i42));
	block_index_space<2> bisa(dimsa), bisb(dimsb);
	block_index_space<4> bisc(dimsc);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<2, double, allocator_t> btb(bisb);
	block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc),
		btc_ref_tmp(bisc);

	//	Load random data for input

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert block tensors to regular tensors

	tensor<2, double, allocator_t> ta(dimsa);
	tensor<2, double, allocator_t> tb(dimsb);
	tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);

	//	Run contraction and compute the reference

	contraction2<2, 2, 0> contr;

	btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc);
	tod_btconv<4>(btc).perform(tc);
	tod_contract2<2, 2, 0>(contr, ta, tb).perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_12() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ij b_lk
	//	Dimensions [ij] = 10, [kl]=20, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_12()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i21, i22;
	i22[0] = 9; i22[1] = 9;
	dimensions<2> dimsa(index_range<2>(i21, i22));
	i22[0] = 19; i22[1] = 19;
	dimensions<2> dimsb(index_range<2>(i21, i22));
	index<4> i41, i42;
	i42[0] = 9; i42[1] = 9; i42[2] = 19; i42[3] = 19;
	dimensions<4> dimsc(index_range<4>(i41, i42));
	block_index_space<2> bisa(dimsa), bisb(dimsb);
	block_index_space<4> bisc(dimsc);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<2, double, allocator_t> btb(bisb);
	block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc),
		btc_ref_tmp(bisc);

	//	Load random data for input

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert block tensors to regular tensors

	tensor<2, double, allocator_t> ta(dimsa);
	tensor<2, double, allocator_t> tb(dimsb);
	tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);

	//	Run contraction and compute the reference

	permutation<4> permc;
	permc.permute(2, 3);
	contraction2<2, 2, 0> contr(permc);

	btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc);
	tod_btconv<4>(btc).perform(tc);
	tod_contract2<2, 2, 0>(contr, ta, tb).perform(tc_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_contract2_test::test_contr_13() throw(libtest::test_exception) {

	//
	//	c_ij = a_kijl b_kl
	//	Dimensions [ij] = 10, [kl]=20, no symmetry
	//

	static const char *testname = "btod_contract2_test::test_contr_13()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i41, i42;
	i42[0] = 19; i42[1] = 9; i42[2] = 9; i42[3] = 19;
	dimensions<4> dimsa(index_range<4>(i41, i42));
	index<2> i21, i22;
	i22[0] = 19; i22[1] = 19;
	dimensions<2> dimsb(index_range<2>(i21, i22));
	i22[0] = 9; i22[1] = 9;
	dimensions<2> dimsc(index_range<2>(i21, i22));
	block_index_space<4> bisa(dimsa);
	block_index_space<2> bisb(dimsb);
	block_index_space<2> bisc(dimsc);

	block_tensor<4, double, allocator_t> bta(bisa);
	block_tensor<2, double, allocator_t> btb(bisb);
	block_tensor<2, double, allocator_t> btc(bisc), btc_ref(bisc),
		btc_ref_tmp(bisc);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	//	Convert block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa);
	tensor<2, double, allocator_t> tb(dimsb);
	tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);

	//	Run contraction and compute the reference

	contraction2<2, 0, 2> contr;
	contr.contract(0, 0);
	contr.contract(3, 1);

	btod_contract2<2, 0, 2> op(contr, bta, btb);
	op.perform(btc);
	tod_btconv<2>(btc).perform(tc);
	tod_contract2<2, 0, 2>(contr, ta, tb).perform(tc_ref);

	//	Compare against reference

	compare_ref<2>::compare(testname, tc, tc_ref, 1e-13);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

