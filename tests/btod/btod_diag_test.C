#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/mask.h>
#include <libtensor/btod/btod_diag.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_diag.h>
#include "btod_diag_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_diag_test::perform() throw(libtest::test_exception) {

	test_zero_1();
	test_zero_2();

	test_nosym_1(false);
	test_nosym_2(false);
	test_nosym_3(false);
	test_nosym_4(false);

	test_nosym_1(true);
	test_nosym_2(true);
	test_nosym_3(true);
	test_nosym_4(true);

	test_sym_1(false);
	test_sym_1(true);

	test_sym_2(false);
	test_sym_2(true);

	test_sym_3(false);
	test_sym_3(true);

	test_sym_4(false);
	test_sym_4(true);

	test_sym_5(false);
	test_sym_5(true);

	test_sym_6(false);
	test_sym_6(true);

	test_sym_7(false);
	test_sym_7(true);
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, zero tensor, one block
 **/
void btod_diag_test::test_zero_1() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_zero_1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<1>().perform(btb);
	bta.set_immutable();

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);

	block_tensor_ctrl<1, double> ctrlb(btb);
	dimensions<1> bidims1 = bis1.get_block_index_dims();
	orbit_list<1, double> olb(ctrlb.req_const_symmetry());
	for (orbit_list<1, double>::iterator ib = olb.begin();
			ib != olb.end(); ib++) {
		orbit<1, double> ob(ctrlb.req_const_symmetry(), olb.get_index(ib));
		abs_index<1> bidx(ob.get_abs_canonical_index(), bidims1);
		if (! ctrlb.req_is_zero_block(bidx.get_index()))
			fail_test(testname, __FILE__, __LINE__, "Unexpected non-zero block.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, zero tensor, multiple blocks
 **/
void btod_diag_test::test_zero_2() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_zero_2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	mask<1> msk1;
	msk1[0] = true;
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	bis1.split(msk1,3); bis1.split(msk1,6);
	bis2.split(msk2,3); bis2.split(msk2,6);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<1>().perform(btb);
	bta.set_immutable();

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);

	block_tensor_ctrl<1, double> ctrlb(btb);
	dimensions<1> bidims1 = bis1.get_block_index_dims();
	orbit_list<1, double> olb(ctrlb.req_const_symmetry());
	for (orbit_list<1, double>::iterator ib = olb.begin();
			ib != olb.end(); ib++) {
		orbit<1, double> ob(ctrlb.req_const_symmetry(), olb.get_index(ib));
		abs_index<1> bidx(ob.get_abs_canonical_index(), bidims1);
		if (! ctrlb.req_is_zero_block(bidx.get_index()))
			fail_test(testname, __FILE__, __LINE__, "Unexpected non-zero block.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
	 single block
 **/
void btod_diag_test::test_nosym_1(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_1(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	dense_tensor<2, double, allocator_t> ta(dims2);
	dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();
	tod_btconv<2>(bta).perform(ta);

	if (add) {
		//  Fill with random data
		btod_random<1>().perform(btb);

		//	Prepare the reference
		tod_btconv<1>(btb).perform(tb_ref);

		tod_diag<2, 2>(ta, msk).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb, 1.0);
	}
	else {
		//	Prepare the reference
		tod_diag<2, 2>(ta, msk).perform(tb_ref);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb);
	}

	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a single diagonal: \f$ b_{ija} = a_{iajb} \f$
 **/
void btod_diag_test::test_nosym_2(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_2(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<3> bis3(dims3);
	block_index_space<4> bis4(dims4);

	mask<3> msk3;
	msk3[2] = true;
	bis3.split(msk3, 6);
	mask<4> msk4;
	msk4[1] = true; msk4[3]=true;
	bis4.split(msk4, 6);

	block_tensor<4, double, allocator_t> bta(bis4);
	block_tensor<3, double, allocator_t> btb(bis3);

	dense_tensor<4, double, allocator_t> ta(dims4);
	dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	permutation<3> pb;
	pb.permute(1,2);

	mask<4> msk;
	msk[1] = true; msk[3] = true;

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<3>().perform(btb);

		//	Prepare the reference
		tod_btconv<3>(btb).perform(tb_ref);

		tod_diag<4, 2>(ta, msk, pb).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, pb).perform(btb, 1.0);

	} else {
		tod_diag<4, 2>(ta, msk, pb).perform(tb_ref);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, pb).perform(btb);
	}

	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
	 multiple blocks
 **/
void btod_diag_test::test_nosym_3(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_3(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	mask<1> msk1;
	msk1[0] = true;
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	bis1.split(msk1,3); bis1.split(msk1,6);
	bis2.split(msk2,3); bis2.split(msk2,6);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	dense_tensor<2, double, allocator_t> ta(dims2);
	dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<1>().perform(btb);

		//	Prepare the reference
		tod_btconv<1>(btb).perform(tb_ref);

		tod_diag<2, 2>(ta, msk).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb, 1.0);
	}
	else {
		tod_diag<2, 2>(ta, msk).perform(tb_ref);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb);
	}

	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ija} = a_{iaja} \f$, non-zero tensor,
	 multiple blocks with permutation
 **/
void btod_diag_test::test_nosym_4(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_4(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<3> bis3(dims3);
	block_index_space<4> bis4(dims4);

	mask<3> msk3;
	msk3[0] = true; msk3[1] = true;
	mask<4> msk4;
	msk4[0] = true; msk4[2] = true;
	bis3.split(msk3,2);
	bis4.split(msk4,2);
	msk3[0] = false; msk3[1] = false; msk3[2] = true;
	msk4[0] = false; msk4[1] = true; msk4[2] = false; msk4[3] = true;
	bis3.split(msk3,3);
	bis4.split(msk4,3);

	block_tensor<4, double, allocator_t> bta(bis4);
	block_tensor<3, double, allocator_t> btb(bis3);

	dense_tensor<4, double, allocator_t> ta(dims4);
	dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	mask<4> msk;
	msk[1] = true; msk[3] = true;

	permutation<3> pb;
	pb.permute(1,2);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);

	if (add) {
		btod_random<3>().perform(btb);

		//	Prepare the reference
		tod_btconv<3>(btb).perform(tb_ref);

		tod_diag<4, 2>(ta, msk, pb).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, pb).perform(btb, 1.0);
	}
	else {
		tod_diag<4, 2>(ta, msk, pb).perform(tb_ref);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, pb).perform(btb);
	}
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, permutational symmetry,
	 multiple blocks
 **/
void btod_diag_test::test_sym_1(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_1(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	mask<1> msk1;
	msk1[0] = true;
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	bis1.split(msk1,3); bis1.split(msk1,6);
	bis2.split(msk2,3); bis2.split(msk2,6);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	dense_tensor<2, double, allocator_t> ta(dims2);
	dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	permutation<2> perm10;
	perm10.permute(0, 1);
	se_perm<2, double> cycle1(perm10, true);
	block_tensor_ctrl<2, double> ctrla(bta);
	ctrla.req_symmetry().insert(cycle1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<1>().perform(btb);

		//	Prepare the reference
		tod_btconv<1>(btb).perform(tb_ref);

		tod_diag<2, 2>(ta, msk).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb, 1.0);
	}
	else {
		tod_diag<2, 2>(ta, msk).perform(tb_ref);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb);
	}
	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ia} = a_{iia} \f$, permutational symmetry,
	 multiple blocks
 **/
void btod_diag_test::test_sym_2(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_2(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 5;
	index<3> i3a, i3b;
	i3b[0] = 10; i3b[1] = 10; i3b[2] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	block_index_space<2> bis2(dims2);
	block_index_space<3> bis3(dims3);

	mask<2> msk2;
	msk2[0] = true; msk2[1] = false;
	mask<3> msk3;
	msk3[0] = true; msk3[1] = true; msk3[2] = false;
	bis2.split(msk2,3); bis2.split(msk2,6);
	bis3.split(msk3,3); bis3.split(msk3,6);
	msk2[0] = false; msk2[1] = true;
	msk3[0] = false; msk3[1] = false; msk3[2] = true;
	bis2.split(msk2,5);
	bis3.split(msk3,5);

	block_tensor<3, double, allocator_t> bta(bis3);
	block_tensor<2, double, allocator_t> btb(bis2);

	dense_tensor<3, double, allocator_t> ta(dims3);
	dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

	permutation<3> perm10;
	perm10.permute(0, 1);
	se_perm<3, double> cycle1(perm10, true);
	block_tensor_ctrl<3, double> ctrla(bta);
	ctrla.req_symmetry().insert(cycle1);

	mask<3> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<3>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<2>().perform(btb);

		//	Prepare the reference
		tod_btconv<2>(btb).perform(tb_ref);


		tod_diag<3, 2>(ta, msk).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<3, 2>(bta, msk).perform(btb, 1.0);
	}
	else {
		tod_diag<3, 2>(ta, msk).perform(tb_ref);

		//	Invoke the operation
		btod_diag<3, 2>(bta, msk).perform(btb);
	}

	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, permutational anti-symmetry,
	 multiple blocks
 **/
void btod_diag_test::test_sym_3(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_3(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	mask<1> msk1;
	msk1[0] = true;
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	bis1.split(msk1,3); bis1.split(msk1,6);
	bis2.split(msk2,3); bis2.split(msk2,6);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	dense_tensor<2, double, allocator_t> ta(dims2);
	dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	permutation<2> perm10;
	perm10.permute(0, 1);
	se_perm<2, double> cycle1(perm10, false);
	block_tensor_ctrl<2, double> ctrla(bta);
	ctrla.req_symmetry().insert(cycle1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<1>().perform(btb);

		//	Prepare the reference
		tod_btconv<1>(btb).perform(tb_ref);

		tod_diag<2, 2>(ta, msk).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb, 1.0);
	}
	else {
		tod_diag<2, 2>(ta, msk).perform(tb_ref);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb);
	}

	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ija} = a_{iaja} \f$, permutational anti-symmetry,
	 multiple blocks
 **/
void btod_diag_test::test_sym_4(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_4(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
	index<4> i4a, i4b;
	i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<3> bis3(dims3);
	block_index_space<4> bis4(dims4);

	mask<3> msk3;
	msk3[0] = true; msk3[1] = true;
	mask<4> msk4;
	msk4[0] = true; msk4[2] = true;
	bis3.split(msk3, 2);
	bis4.split(msk4, 2);
	msk3[0] = false; msk3[1] = false; msk3[2] = true;
	msk4[0] = false; msk4[1] = true; msk4[2] = false; msk4[3] = true;
	bis3.split(msk3,3);
	bis4.split(msk4,3);

	block_tensor<4, double, allocator_t> bta(bis4);
	block_tensor<3, double, allocator_t> btb(bis3);

	dense_tensor<4, double, allocator_t> ta(dims4);
	dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	permutation<4> perm20;
	perm20.permute(0, 2);
	se_perm<4, double> cycle1(perm20, false);
	block_tensor_ctrl<4, double> ctrla(bta);
	ctrla.req_symmetry().insert(cycle1);

	mask<4> msk;
	msk[1] = true; msk[3] = true;

	permutation<3> pb;
	pb.permute(1,2);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<3>().perform(btb);

		//	Prepare the reference
		tod_btconv<3>(btb).perform(tb_ref);

		tod_diag<4, 2>(ta, msk, pb).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, pb).perform(btb, 1.0);
	}
	else {
		tod_diag<4, 2>(ta, msk, pb).perform(tb_ref);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, pb).perform(btb);
	}
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_{iaj} = a_{iaja} \f$,
		permutational symmetry, multiple blocks
 **/
void btod_diag_test::test_sym_5(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_5(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 7; i3b[1] = 7; i3b[2] = 10;
	index<4> i4a, i4b;
	i4b[0] = 7; i4b[1] = 10; i4b[2] = 7; i4b[3] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<3> bis3(dims3);
	block_index_space<4> bis4(dims4);

	mask<3> msk3;
	msk3[0] = true; msk3[1] = true;
	mask<4> msk4;
	msk4[0] = true; msk4[2] = true;
	bis3.split(msk3,2); bis3.split(msk3, 5);
	bis4.split(msk4,2); bis4.split(msk4, 5);
	msk3[0] = false; msk3[1] = false; msk3[2] = true;
	msk4[0] = false; msk4[1] = true; msk4[2] = false; msk4[3] = true;
	bis3.split(msk3, 5);
	bis4.split(msk4, 5);

	block_tensor<4, double, allocator_t> bta(bis4);
	block_tensor<3, double, allocator_t> btb(bis3);

	dense_tensor<4, double, allocator_t> ta(dims4);
	dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	{
	se_perm<4, double> cycle1(permutation<4>().permute(0, 2).permute(1, 3), true);
	block_tensor_ctrl<4, double> ctrla(bta);
	ctrla.req_symmetry().insert(cycle1);
	}

	mask<4> msk;
	msk[1] = true; msk[3] = true;
	permutation<3> perm;
	perm.permute(1, 2);

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<3>().perform(btb);

		//	Prepare the reference
		tod_btconv<3>(btb).perform(tb_ref);

		tod_diag<4, 2>(ta, msk, perm).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, perm).perform(btb, 1.0);
	}
	else {
		tod_diag<4, 2>(ta, msk, perm).perform(tb_ref);

		//	Invoke the operation
		btod_diag<4, 2>(bta, msk, perm).perform(btb);
	}
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ijk} = a_{ikjk} \f$,
		permutational anti-symmetry, multiple blocks
 **/
void btod_diag_test::test_sym_6(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_6(bool)";

	typedef std_allocator<double> allocator_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 10; i3b[1] = 10; i3b[2] = 10;
	index<4> i4a, i4b;
	i4b[0] = 10; i4b[1] = 10; i4b[2] = 10; i4b[3] = 10;
	dimensions<3> dims3(index_range<3>(i3a, i3b));
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<3> bis3(dims3);
	block_index_space<4> bis4(dims4);

	mask<3> msk3;
	msk3[0] = true; msk3[1] = true; msk3[2] = true;
	mask<4> msk4;
	msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
	bis3.split(msk3, 2);
	bis3.split(msk3, 4);
	bis3.split(msk3, 8);
	bis4.split(msk4, 2);
	bis4.split(msk4, 4);
	bis4.split(msk4, 8);

	block_tensor<4, double, allocator_t> bta(bis4);
	block_tensor<3, double, allocator_t> btb(bis3);

	dense_tensor<4, double, allocator_t> ta(dims4);
	dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	block_tensor_ctrl<4, double> ctrla(bta);

	ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
		permute(0, 1), true));
	ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
		permute(2, 3), true));

	mask<4> msk;
	msk[1] = true; msk[3] = true;

	//	Fill in random data
	btod_random<4>().perform(bta);
	btod_random<3>().perform(btb);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<3>(btb).perform(tb_ref);

	//	Invoke the operation
	if(add) {
		btod_diag<4, 2>(bta, msk).perform(btb, 1.0);
		tod_diag<4, 2>(ta, msk).perform(tb_ref, 1.0);
	} else {
		btod_diag<4, 2>(bta, msk).perform(btb);
		tod_diag<4, 2>(ta, msk).perform(tb_ref);
	}
	tod_btconv<3>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
		multiple blocks, label symmetry
 **/
void btod_diag_test::test_sym_7(bool add) throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_7(bool)";

	typedef std_allocator<double> allocator_t;

	bool need_erase = true;
	const char *pgtid = "point_group_cs";

	try {

	point_group_table::label_t ap = 0, app = 1;
	std::vector<std::string> irnames(2);
	irnames[0] = "Ap"; irnames[1] = "App";
	point_group_table cs(pgtid, irnames, irnames[0]);
	cs.add_product(ap, ap, ap);
	cs.add_product(ap, app, app);
	cs.add_product(app, ap, app);
	cs.add_product(app, app, ap);
	cs.check();
	product_table_container::get_instance().add(cs);

	{

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<1> bis1(dims1);
	block_index_space<2> bis2(dims2);

	mask<1> msk1;
	msk1[0] = true;
	mask<2> msk2;
	msk2[0] = true; msk2[1] = true;
	bis1.split(msk1, 3); bis1.split(msk1, 6);
	bis2.split(msk2, 3); bis2.split(msk2, 6);

	se_label<2, double> elem1(bis2.get_block_index_dims(), pgtid);
	block_labeling<2> &bl1 = elem1.get_labeling();
	bl1.assign(msk2, 0, ap);
	bl1.assign(msk2, 1, ap);
	bl1.assign(msk2, 2, app);
	elem1.set_rule(ap);

	block_tensor<2, double, allocator_t> bta(bis2);
	block_tensor<1, double, allocator_t> btb(bis1);

	{
		block_tensor_ctrl<2, double> ca(bta);
		ca.req_symmetry().insert(elem1);
	}

	dense_tensor<2, double, allocator_t> ta(dims2);
	dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);

	if (add) {
		//	Fill in random data
		btod_random<1>().perform(btb);

		//	Prepare the reference
		tod_btconv<1>(btb).perform(tb_ref);

		tod_diag<2, 2>(ta, msk).perform(tb_ref, 1.0);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb, 1.0);
	}
	else {
		tod_diag<2, 2>(ta, msk).perform(tb_ref);

		//	Invoke the operation
		btod_diag<2, 2>(bta, msk).perform(btb);
	}

	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	}

	need_erase = false;
	product_table_container::get_instance().erase(pgtid);

	} catch(exception &e) {
		if(need_erase) {
			product_table_container::get_instance().erase(pgtid);
		}
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
