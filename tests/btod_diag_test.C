#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/mask.h>
#include <libtensor/btod/btod_diag.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_diag.h>
#include "btod_diag_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_diag_test::perform() throw(libtest::test_exception) {

	test_zero_1();
	test_zero_2();

	test_nosym_1();
	test_nosym_2();
	test_nosym_3();
	test_nosym_4();

	test_sym_1();
	test_sym_2();
	test_sym_3();
	test_sym_4();

}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, zero tensor, one block
 **/
void btod_diag_test::test_zero_1() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_zero_1()";

	typedef libvmm::std_allocator<double> allocator_t;

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
		index<1> bidx;
		bidims1.abs_index(ob.get_abs_canonical_index(), bidx);
		if (! ctrlb.req_is_zero_block(bidx))
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

	typedef libvmm::std_allocator<double> allocator_t;

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
		index<1> bidx;
		bidims1.abs_index(ob.get_abs_canonical_index(), bidx);
		if (! ctrlb.req_is_zero_block(bidx))
			fail_test(testname, __FILE__, __LINE__, "Unexpected non-zero block.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
	 single block
 **/
void btod_diag_test::test_nosym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_1()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<2, double, allocator_t> ta(dims2);
	tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	tod_diag<2, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);
	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract a single diagonal: \f$ b_{ija} = a_{iajb} \f$
 **/
void btod_diag_test::test_nosym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_2()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<4, double, allocator_t> ta(dims4);
	tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

	permutation<3> pb;
	pb.permute(1,2);

	mask<4> msk;
	msk[1] = true; msk[3] = true;

	//	Fill in random data
	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<4>(bta).perform(ta);
	tod_diag<4, 2>(ta, msk, pb).perform(tb_ref);

	//	Invoke the operation
	btod_diag<4, 2>(bta, msk, pb).perform(btb);
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
void btod_diag_test::test_nosym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_3()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<2, double, allocator_t> ta(dims2);
	tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

	mask<2> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<2>(bta).perform(ta);
	tod_diag<2, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);
	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ia} = a_{iia} \f$, non-zero tensor,
	 single block
 **/
void btod_diag_test::test_nosym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_nosym_4()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<3, double, allocator_t> ta(dims3);
	tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

	mask<3> msk;
	msk[0] = true; msk[1] = true;

	//	Fill in random data
	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Prepare the reference
	tod_btconv<3>(bta).perform(ta);
	tod_diag<3, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<3, 2>(bta, msk).perform(btb);
	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Extract diagonal: \f$ b_i = a_{ii} \f$, permutational symmetry,
	 multiple blocks
 **/
void btod_diag_test::test_sym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_1()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<2, double, allocator_t> ta(dims2);
	tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

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
	tod_diag<2, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);
	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ia} = a_{iia} \f$, permutational symmetry,
	 multiple bloacks
 **/
void btod_diag_test::test_sym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_2()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<3, double, allocator_t> ta(dims3);
	tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

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
	tod_diag<3, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<3, 2>(bta, msk).perform(btb);
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
void btod_diag_test::test_sym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_3()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<2, double, allocator_t> ta(dims2);
	tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

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
	tod_diag<2, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<2, 2>(bta, msk).perform(btb);
	tod_btconv<1>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Extract diagonal: \f$ b_{ia} = a_{iia} \f$, non-zero tensor, single block
 **/
void btod_diag_test::test_sym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_diag_test::test_sym_4()";

	typedef libvmm::std_allocator<double> allocator_t;

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

	tensor<3, double, allocator_t> ta(dims3);
	tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

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
	tod_diag<3, 2>(ta, msk).perform(tb_ref);

	//	Invoke the operation
	btod_diag<3, 2>(bta, msk).perform(btb);
	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference
	compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
