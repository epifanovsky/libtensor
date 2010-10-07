#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_symmetrize.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/tod/tod_btconv.h>
#include "btod_symmetrize_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_symmetrize_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5(false);
	test_5(true);
	test_6(false, false, false, false);
	test_6(false, false, false, true);
	test_6(false, false, true, false);
	test_6(false, false, true, true);
	test_6(false, true, false, false);
	test_6(false, true, false, true);
	test_6(false, true, true, false);
	test_6(false, true, true, true);
	test_6(true, false, false, false);
	test_6(true, false, false, true);
	test_6(true, false, true, false);
	test_6(true, false, true, true);
	test_6(true, true, false, false);
	test_6(true, true, false, true);
	test_6(true, true, true, false);
	test_6(true, true, true, true);
	test_7();
	test_8();
}


/**	\test Symmetrization of a non-symmetric 2-index block %tensor
 **/
void btod_symmetrize_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Fill in random input

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<2>(bta).perform(ta);
	tod_add<2> refop(ta);
	refop.add_op(ta, permutation<2>().permute(0, 1), 1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<2> op_copy(bta);
	btod_symmetrize<2>(op_copy, 0, 1, true).perform(btb);

	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<2, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<2, double> ctrlb(btb);
		so_copy<2, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), true));

	compare_ref<2>::compare(testname, symb, symb_ref);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Anti-symmetrization of a non-symmetric 2-index block %tensor
 **/
void btod_symmetrize_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Fill in random input

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<2>(bta).perform(ta);
	tod_add<2> refop(ta);
	refop.add_op(ta, permutation<2>().permute(0, 1), -1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<2> op_copy(bta);
	btod_symmetrize<2>(op_copy, 0, 1, false).perform(btb);

	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<2, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<2, double> ctrlb(btb);
		so_copy<2, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), false));

	compare_ref<2>::compare(testname, symb, symb_ref);

	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Anti-symmetrization of S(-)2*C1*C1 to S(-)2*S(-)2
		in a 4-index block %tensor
 **/
void btod_symmetrize_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 20; i2[1] = 10; i2[2] = 20; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m1, m2;
	m1[0] = true; m2[1] = true; m1[2] = true; m2[3] = true;
	bis.split(m1, 5);
	bis.split(m1, 10);
	bis.split(m2, 2);
	bis.split(m2, 5);

	block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Set up initial symmetry and fill in random input

	{
		block_tensor_ctrl<4, double> ctrla(bta);
		ctrla.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(0, 2), false));
	}
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<4>(bta).perform(ta);
	tod_add<4> refop(ta);
	refop.add_op(ta, permutation<4>().permute(1, 3), -1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<4> op_copy(bta);
	btod_symmetrize<4>(op_copy, 1, 3, false).perform(btb);

	tod_btconv<4>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<4, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<4, double> ctrlb(btb);
		so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 2), false));
	symb_ref.insert(se_perm<4, double>(
		permutation<4>().permute(1, 3), false));

	compare_ref<4>::compare(testname, symb, symb_ref);

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of S2*S2 to S2*C1*C1 in a 4-index block %tensor
 **/
void btod_symmetrize_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;

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
			permutation<4>().permute(0, 1), true));
		ctrla.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(2, 3), true));
	}
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<4>(bta).perform(ta);
	tod_add<4> refop(ta);
	refop.add_op(ta, permutation<4>().permute(0, 2), 1.0);
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<4> op_copy(bta);
	btod_symmetrize<4>(op_copy, 0, 2, true).perform(btb);

	tod_btconv<4>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<4, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<4, double> ctrlb(btb);
		so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 2), true));

	compare_ref<4>::compare(testname, symb, symb_ref);

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of two pairs of indexes in a non-symmetric
		4-index block %tensor
 **/
void btod_symmetrize_test::test_5(bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_5(bool)";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 20; i2[1] = 20; i2[2] = 20; i2[3] = 20;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 5);
	bis.split(m, 10);
	bis.split(m, 15);

	block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

	//	Fill in random input

	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare reference data

	tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<4>(bta).perform(ta);
	tod_add<4> refop(ta);
	refop.add_op(ta, permutation<4>().permute(0, 2).permute(1, 3),
			(symm ? 1.0 : -1.0));
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<4> op_copy(bta);
	btod_symmetrize<4>(op_copy, permutation<4>().permute(0, 2).
		permute(1, 3), symm).perform(btb);

	tod_btconv<4>(btb).perform(tb);

	//	Compare against the reference: symmetry and data

	symmetry<4, double> symb(bis), symb_ref(bis);
	{
		block_tensor_ctrl<4, double> ctrlb(btb);
		so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
	}
	symb_ref.insert(se_perm<4, double>(
		permutation<4>().permute(0, 2).permute(1, 3), symm));

	compare_ref<4>::compare(testname, symb, symb_ref);

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test (Anti-)Symmetrization of two indexes in a non-symmetric
		2-dim block %tensor with se_label, se_part
 **/
void btod_symmetrize_test::test_6(bool symm, bool label,
		bool part, bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_symmetrize_test::test_6(" << symm << ", "
			<< label << ", " << part << ", " << doadd << ")";
	std::string tns = tnss.str();

	typedef libvmm::std_allocator<double> allocator_t;

	if (label) {
		point_group_table pg(tns, 2);
		pg.add_product(0, 0, 0);
		pg.add_product(0, 1, 1);
		pg.add_product(1, 1, 0);

		product_table_container::get_instance().add(pg);
	}

	try {

	index<2> i1, i2;
	i2[0] = 11; i2[1] = 11;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 6);
	bis.split(m, 8);

	permutation<2> p;
	p.permute(0, 1);

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	symmetry<2, double> sym_ref(bis);

	// setup symmetry
	{
	block_tensor_ctrl<2, double> ca(bta), cb(btb);

	se_perm<2, double> se10(p, symm);
	cb.req_symmetry().insert(se10);
	sym_ref.insert(se10);

	if (label) {
		se_label<2, double> sl(bis.get_block_index_dims(), tns);
		sl.assign(m, 0, 0);
		sl.assign(m, 1, 1);
		sl.assign(m, 2, 0);
		sl.assign(m, 3, 1);
		sl.add_target(1);
		ca.req_symmetry().insert(sl);
		cb.req_symmetry().insert(sl);
		sym_ref.insert(sl);
	}

	if (part) {
		se_part<2, double> sp(bis, m, 2);
		index<2> i00, i01, i10, i11;
		i10[0] = 1; i01[1] = 1;
		i11[0] = 1; i11[1] = 1;
		sp.add_map(i00, i11);
		ca.req_symmetry().insert(sp);
		cb.req_symmetry().insert(sp);
		sym_ref.insert(sp);
	}
	}
	//	Fill in random input

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();

	//	Prepare reference data

	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	tod_btconv<2>(bta).perform(ta);
	tod_add<2> refop(ta);
	refop.add_op(ta, p, (symm ? 1.0 : -1.0));
	if (doadd) {
		tod_btconv<2>(btb).perform(tb_ref);
		refop.perform(tb_ref, 0.25);
	}
	else {
		refop.perform(tb_ref);
	}

	//	Run the symmetrization operation

	btod_copy<2> op_copy(bta);
	if (doadd) btod_symmetrize<2>(op_copy, p, symm).perform(btb, 0.25);
	else btod_symmetrize<2>(op_copy, p, symm).perform(btb);

	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference: symmetry and data
	block_tensor_ctrl<2, double> ctrlb(btb);

	compare_ref<2>::compare(tns.c_str(), ctrlb.req_const_symmetry(), sym_ref);

	compare_ref<2>::compare(tns.c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		if (label) product_table_container::get_instance().erase(tns);

		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		if (label) product_table_container::get_instance().erase(tns);

		throw;
	}
	if (label) product_table_container::get_instance().erase(tns);

}

/**	\test Anti-symmetrization of a non-symmetric 3-index block %tensor
		over three indexes
 **/
void btod_symmetrize_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_7()";

	typedef libvmm::std_allocator<double> allocator_t;

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
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<3> op_copy(bta);
	btod_symmetrize<3>(op_copy, 0, 1, 2, false).perform(btb);

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
void btod_symmetrize_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "btod_symmetrize_test::test_8()";

	typedef libvmm::std_allocator<double> allocator_t;

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
	refop.perform(tb_ref);

	//	Run the symmetrization operation

	btod_copy<3> op_copy(bta);
	btod_symmetrize<3>(op_copy, 0, 1, 2, true).perform(btb);

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


} // namespace libtensor

