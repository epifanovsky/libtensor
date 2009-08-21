#include <sstream>
#include <libvmm.h>
#include <libtensor.h>
#include "btod_contract2_test.h"

namespace libtensor {


void btod_contract2_test::perform() throw(libtest::test_exception) {
	typedef libvmm::std_allocator<double> allocator;
	typedef block_tensor<2,double,allocator> block_tensor2;

	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	i2[0] = 1; i2[1] = 1;
	dimensions<2> dims_ij(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai), ij(dims_ij);

	block_tensor2 bt1(ia), bt2(ai), bt3(ij);
	permutation<2> p;

	contraction2<1,1,1> contr(p);
	contr.contract(1,0);

	btod_contract2<1,1,1> operation(contr,bt1,bt2);
	operation.perform(bt3,0.1);

	operation.perform(bt3);

	test_bis_1();
	test_bis_2();
	test_sym_1();
	test_sym_2();
}


void btod_contract2_test::test_bis_1() throw(libtest::test_exception) {

	//
	//	c_ijkl = a_ijqr b_klpq
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

	if(op.get_symmetry().get_num_elements() != sym_ref.get_num_elements()) {
		std::ostringstream ss;
		ss << "Incorrect number of symmetry elements: "
			<< op.get_symmetry().get_num_elements() << " vs. "
			<< sym_ref.get_num_elements() << " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	size_t nelem = sym_ref.get_num_elements();
	for(size_t ielem = 0; ielem < nelem; ielem++) {
		if(!op.get_symmetry().contains_element(
			sym_ref.get_element(ielem))) {

			fail_test(testname, __FILE__, __LINE__,
				"Reference symmetry element not found.");
		}
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

	if(op.get_symmetry().get_num_elements() != 0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect number of symmetry elements.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

