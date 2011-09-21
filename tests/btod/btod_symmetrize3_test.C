#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_symmetrize3.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/tod/tod_btconv.h>
#include "btod_symmetrize3_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_symmetrize3_test::perform() throw(libtest::test_exception) {

	allocator<double>::vmm().init(16, 16, 16777216, 16777216);

	try {

		test_1();
		test_2();
		test_3();
		test_4();
		test_5();
		test_6();
		test_7();

	} catch(...) {
		allocator<double>::vmm().shutdown();
		throw;
	}

	allocator<double>::vmm().shutdown();
}


/**	\test Symmetrization of a non-symmetric 3-index block %tensor
		over three indexes
 **/
void btod_symmetrize3_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10;
	dimensions<3> dims(index_range<3>(i1, i2));
	block_index_space<3> bis(dims);
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Fill in random input

	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<3>(bta).perform(ta);
	tod_add<3> refop(ta);
	refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
	refop.perform(tb_ref);

	symmetry<3, double> symb(bis), symb_ref(bis);
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1), true));
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 2), true));

	//	Run the symmetrization operation

	btod_copy<3> op_copy(bta);
	btod_symmetrize3<3> op_sym(op_copy, 0, 1, 2, true);

	compare_ref<3>::compare(testname, op_sym.get_symmetry(), symb_ref);

	op_sym.perform(btb);
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	{
		block_tensor_ctrl<3, double> ctrlb(btb);
		so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
	}

	compare_ref<3>::compare(testname, symb, symb_ref);
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Anti-symmetrization of a non-symmetric 3-index block %tensor
		over three indexes
 **/
void btod_symmetrize3_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10;
	dimensions<3> dims(index_range<3>(i1, i2));
	block_index_space<3> bis(dims);
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Fill in random input

	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<3>(bta).perform(ta);
	tod_add<3> refop(ta);
	refop.add_op(ta, permutation<3>().permute(0, 1), -1.0);
	refop.add_op(ta, permutation<3>().permute(0, 2), -1.0);
	refop.add_op(ta, permutation<3>().permute(1, 2), -1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<3> op_copy(bta);
	btod_symmetrize3<3>(op_copy, 0, 1, 2, false).perform(btb);

	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<3, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<3, double> ctrlb(btb);
		so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1), false));
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 2), false));

	compare_ref<3>::compare(testname, symb, symb_ref);

	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of a 3-index block %tensor with S(+)2*C1
		over three indexes
 **/
void btod_symmetrize3_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10;
	dimensions<3> dims(index_range<3>(i1, i2));
	block_index_space<3> bis(dims);
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Set up initial symmetry and fill in random input

	{
		block_tensor_ctrl<3, double> ctrla(bta);
		ctrla.req_symmetry().insert(se_perm<3, double>(
			permutation<3>().permute(1, 2), true));
	}
	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<3>(bta).perform(ta);
	tod_add<3> refop(ta);
	refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<3> op_copy(bta);
	btod_symmetrize3<3>(op_copy, 0, 1, 2, true).perform(btb);

	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<3, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<3, double> ctrlb(btb);
		so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1), true));
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1).permute(1, 2), true));

	compare_ref<3>::compare(testname, symb, symb_ref);

	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of a 4-index block %tensor with S(+)2*C1
		over three indexes
 **/
void btod_symmetrize3_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_4()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Set up initial symmetry and fill in random input

	{
		block_tensor_ctrl<4, double> ctrla(bta);
		ctrla.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(2, 3), true));
	}
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<4>(bta).perform(ta);
	tod_add<4> refop(ta);
	refop.add_op(ta, permutation<4>().permute(0, 1), 1.0);
	refop.add_op(ta, permutation<4>().permute(0, 2), 1.0);
	refop.add_op(ta, permutation<4>().permute(1, 2), 1.0);
	refop.add_op(ta, permutation<4>().permute(0, 1).permute(0, 2), 1.0);
	refop.add_op(ta, permutation<4>().permute(0, 1).permute(1, 2), 1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<4> op_copy(bta);
	btod_symmetrize3<4>(op_copy, 0, 1, 2, true).perform(btb);
	tod_btconv<4>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<4, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<4, double> ctrlb(btb);
		so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	symb_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 1).permute(1, 2), true));

	compare_ref<4>::compare(testname, symb, symb_ref);

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of a 3-index block %tensor with partitions
		over three indexes
 **/
void btod_symmetrize3_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_5()";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9;
	dimensions<3> dims(index_range<3>(i1, i2));
	block_index_space<3> bis(dims);
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	bis.split(m, 7);

	block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Set up initial symmetry and fill in random input

	{
		index<3> i000, i001, i010, i011, i100, i101, i110, i111;
		i110[0] = 1; i110[1] = 1; i001[2] = 1;
		i101[0] = 1; i010[1] = 1; i101[2] = 1;
		i100[0] = 1; i011[1] = 1; i011[2] = 1;
		i111[0] = 1; i111[1] = 1; i111[2] = 1;
		block_tensor_ctrl<3, double> ctrla(bta);
		se_part<3, double> p(bis, m, 2);
		p.add_map(i000, i111, true);
		p.add_map(i001, i110, true);
		p.add_map(i010, i101, true);
		p.add_map(i011, i100, true);
		ctrla.req_symmetry().insert(p);
	}
	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<3>(bta).perform(ta);
	tod_add<3> refop(ta);
	refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
	refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
	refop.perform(tb_ref);

	symmetry<3, double> symb(bis), symb_ref(bis);
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 1), true));
	symb_ref.insert(se_perm<3, double>(
		permutation<3>().permute(0, 2), true));
	{
		index<3> i000, i001, i010, i011, i100, i101, i110, i111;
		i110[0] = 1; i110[1] = 1; i001[2] = 1;
		i101[0] = 1; i010[1] = 1; i101[2] = 1;
		i100[0] = 1; i011[1] = 1; i011[2] = 1;
		i111[0] = 1; i111[1] = 1; i111[2] = 1;
		block_tensor_ctrl<3, double> ctrla(bta);
		se_part<3, double> p(bis, m, 2);
		p.add_map(i000, i111, true);
		p.add_map(i001, i110, true);
		p.add_map(i010, i101, true);
		p.add_map(i011, i100, true);
		symb_ref.insert(p);
	}

	//	Run the symmetrization operation

	btod_copy<3> op_copy(bta);
	btod_symmetrize3<3> op_sym(op_copy, 0, 1, 2, true);

	compare_ref<3>::compare(testname, op_sym.get_symmetry(), symb_ref);

	op_sym.perform(btb);
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	{
		block_tensor_ctrl<3, double> ctrlb(btb);
		so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
	}

	compare_ref<3>::compare(testname, symb, symb_ref);
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Double anti-symmetrization of a 6-index block %tensor with
		S(-)2*C1*C1*S(-)2 over three indexes
 **/
void btod_symmetrize3_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_6()";

	typedef std_allocator<double> allocator_t;

	try {

	index<6> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 7; i2[4] = 7; i2[5] = 7;
	dimensions<6> dims(index_range<6>(i1, i2));
	block_index_space<6> bis(dims);

	block_tensor<6, double, allocator_t> bta(bis), btb(bis);

	//	Set up initial symmetry and fill in random input

	{
		block_tensor_ctrl<6, double> ctrla(bta);
		ctrla.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(0, 1), false));
		ctrla.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(4, 5), false));
	}

	btod_random<6>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<6, double, allocator_t> ta(dims), ta1(dims), tb(dims),
		tb_ref(dims);
	tod_btconv<6>(bta).perform(ta);
	tod_add<6> refop1(ta);
	refop1.add_op(ta, permutation<6>().permute(0, 1), -1.0);
	refop1.add_op(ta, permutation<6>().permute(0, 2), -1.0);
	refop1.add_op(ta, permutation<6>().permute(1, 2), -1.0);
	refop1.add_op(ta, permutation<6>().permute(0, 1).permute(0, 2), 1.0);
	refop1.add_op(ta, permutation<6>().permute(0, 1).permute(1, 2), 1.0);
	refop1.perform(ta1);
	tod_add<6> refop2(ta1);
	refop2.add_op(ta1, permutation<6>().permute(3, 4), -1.0);
	refop2.add_op(ta1, permutation<6>().permute(3, 5), -1.0);
	refop2.add_op(ta1, permutation<6>().permute(4, 5), -1.0);
	refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(3, 5), 1.0);
	refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(4, 5), 1.0);
	refop2.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<6> op_copy(bta);
	btod_symmetrize3<6> op_sym3(op_copy, 0, 1, 2, false);
	btod_symmetrize3<6>(op_sym3, 3, 4, 5, false).perform(btb);

	tod_btconv<6>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<6, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<6, double> ctrlb(btb);
		so_copy<6, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(0, 1), false));
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(0, 1).permute(1, 2), false));
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(3, 4), false));
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(3, 4).permute(4, 5), false));

	compare_ref<6>::compare(testname, symb, symb_ref);

	compare_ref<6>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Double anti-symmetrization of a 6-index block %tensor with
		S(-)2*C1*C1*S(-)2 over three indexes (additive)
 **/
void btod_symmetrize3_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize3_test::test_7()";

	typedef std_allocator<double> allocator_t;

	try {

	index<6> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 7; i2[4] = 7; i2[5] = 7;
	dimensions<6> dims(index_range<6>(i1, i2));
	block_index_space<6> bis(dims);

	block_tensor<6, double, allocator_t> bta(bis), btb(bis);

	//	Set up initial symmetry and fill in random input

	{
		block_tensor_ctrl<6, double> ctrla(bta), ctrlb(btb);
		ctrla.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(0, 1), false));
		ctrla.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(4, 5), false));
		ctrlb.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(0, 1), false));
		ctrlb.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(1, 2), false));
		ctrlb.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(3, 4), false));
		ctrlb.req_symmetry().insert(se_perm<6, double>(
			permutation<6>().permute(4, 5), false));
	}

	btod_random<6>().perform(bta);
	btod_random<6>().perform(btb);
	bta.set_immutable();

	//	Prepare reference data

	tensor<6, double, allocator_t> ta(dims), ta1(dims), tb(dims),
		tb_ref(dims);
	tod_btconv<6>(bta).perform(ta);
	tod_btconv<6>(btb).perform(tb_ref);
	tod_add<6> refop1(ta);
	refop1.add_op(ta, permutation<6>().permute(0, 1), -1.0);
	refop1.add_op(ta, permutation<6>().permute(0, 2), -1.0);
	refop1.add_op(ta, permutation<6>().permute(1, 2), -1.0);
	refop1.add_op(ta, permutation<6>().permute(0, 1).permute(0, 2), 1.0);
	refop1.add_op(ta, permutation<6>().permute(0, 1).permute(1, 2), 1.0);
	refop1.perform(ta1);
	tod_add<6> refop2(ta1);
	refop2.add_op(ta1, permutation<6>().permute(3, 4), -1.0);
	refop2.add_op(ta1, permutation<6>().permute(3, 5), -1.0);
	refop2.add_op(ta1, permutation<6>().permute(4, 5), -1.0);
	refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(3, 5), 1.0);
	refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(4, 5), 1.0);
	refop2.perform(tb_ref, 2.0);

	//	Run the symmetrization operation

	btod_copy<6> op_copy(bta);
	btod_symmetrize3<6> op_sym3(op_copy, 0, 1, 2, false);
	btod_symmetrize3<6>(op_sym3, 3, 4, 5, false).perform(btb, 2.0);

	tod_btconv<6>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<6, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<6, double> ctrlb(btb);
		so_copy<6, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(0, 1), false));
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(0, 1).permute(1, 2), false));
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(3, 4), false));
	symb_ref.insert(se_perm<6, double>(
		permutation<6>().permute(3, 4).permute(4, 5), false));

	compare_ref<6>::compare(testname, symb, symb_ref);

	compare_ref<6>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor

