#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include "compare_ref.h"
#include "btod_copy_test.h"

namespace libtensor {


void btod_copy_test::perform() throw(libtest::test_exception) {

	test_zero_1();
	test_zero_2();

	test_nosym_1();
	test_nosym_2();
	test_nosym_3();
	test_nosym_4();
	test_sym_1();
	test_sym_2();

	test_add_nosym_1();

	test_2();
	test_3();
	test_4();
	test_dir_1();
	test_dir_2();
	test_dir_3();
	test_dir_4();
}


/**	\test \f$ b_{ij} = a_{ij} \f$, zero tensor
 **/
void btod_copy_test::test_zero_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_zero_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the output with random data

	btod_random<2>().perform(btb);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta).perform(btb);

	//	The set of non-zero blocks in the output must be empty now

	block_tensor_ctrl<2, double> btb_ctrl(btb);
	orbit_list<2, double> orblst(btb_ctrl.req_symmetry());
	orbit_list<2, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {
		orbit<2, double> orb(btb_ctrl.req_symmetry(),
			orblst.get_index(iorbit));
		index<2> blkidx;
		bidims.abs_index(orb.get_abs_canonical_index(), blkidx);
		if(!btb_ctrl.req_is_zero_block(blkidx)) {
			fail_test(testname, __FILE__, __LINE__,
				"All blocks are expected to be empty.");
		}
	}

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = a_{ij} \f$, zero tensor
 **/
void btod_copy_test::test_zero_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_zero_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 3);
	bis.split(msk1, 6);
	bis.split(msk2, 5);
	dimensions<2> bidims(bis.get_block_index_dims());
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the output with random data

	btod_random<2>().perform(btb);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta).perform(btb);

	//	The set of non-zero blocks in the output must be empty now

	block_tensor_ctrl<2, double> btb_ctrl(btb);
	orbit_list<2, double> orblst(btb_ctrl.req_symmetry());
	orbit_list<2, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {
		orbit<2, double> orb(btb_ctrl.req_symmetry(),
			orblst.get_index(iorbit));
		index<2> blkidx;
		bidims.abs_index(orb.get_abs_canonical_index(), blkidx);
		if(!btb_ctrl.req_is_zero_block(blkidx)) {
			fail_test(testname, __FILE__, __LINE__,
				"All blocks are expected to be empty.");
		}
	}

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = a_{ij} \f$, no symmetry, no blocks
 **/
void btod_copy_test::test_nosym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_nosym_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the input with random data

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta).perform(btb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);

	compare_ref<2>::compare(testname, tb, ta, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = 2 a_{ji} \f$, no symmetry, no blocks
 **/
void btod_copy_test::test_nosym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_nosym_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	permutation<2> perm10;
	perm10.permute(0, 1);

	//	Fill the input with random data

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta, perm10, 2.0).perform(btb);

	//	Create the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, perm10, 2.0).perform(tb_ref);

	//	Compare against the reference

	tod_btconv<2>(btb).perform(tb);
	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = a_{ij} \f$, no symmetry, 3 blocks along each
		direction
 **/
void btod_copy_test::test_nosym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_nosym_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the input with random data

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta).perform(btb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);

	compare_ref<2>::compare(testname, tb, ta, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = 2 a_{ji} \f$, no symmetry, 3 blocks along each
		direction
 **/
void btod_copy_test::test_nosym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_nosym_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	permutation<2> perm10;
	perm10.permute(0, 1);

	//	Fill the input with random data

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta, perm10, 2.0).perform(btb);

	//	Create the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, perm10, 2.0).perform(tb_ref);

	//	Compare against the reference

	tod_btconv<2>(btb).perform(tb);
	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = a_{ij} \f$, perm symmetry, 3 blocks along each
		direction
 **/
void btod_copy_test::test_sym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_sym_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the input with random data

	permutation<2> perm10;
	perm10.permute(0, 1);
	se_perm<2, double> cycle1(perm10, true);
	block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta).perform(btb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);

	compare_ref<2>::compare(testname, tb, ta, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = -2 a_{ji} \f$, perm antisymmetry,
		3 blocks along each direction
 **/
void btod_copy_test::test_sym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_sym_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 7);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the input with random data

	permutation<2> perm10;
	perm10.permute(0, 1);
	se_perm<2, double> cycle1(perm10, false);
	block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<2>(bta, perm10, -2.0).perform(btb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, 2.0).perform(tb_ref);
	tod_btconv<2>(btb).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ijk} = 0.3 a_{kji} \f$, perm symmetry, 3 blocks along each
		direction
 **/
void btod_copy_test::test_sym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_sym_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<3> i1a, i2a;
	i2a[0] = 5; i2a[1] = 10; i2a[2] = 10;
	index<3> i1b, i2b;
	i2b[0] = 10; i2b[1] = 10; i2b[2] = 5;
	dimensions<3> dima(index_range<3>(i1a, i2a));
	dimensions<3> dimb(index_range<3>(i1b, i2b));
	block_index_space<3> bisa(dima), bisb(dimb);
	mask<3> ma1, ma2, mb1, mb2;
	ma2[0] = true; ma1[1] = true; ma1[2] = true;
	bisa.split(ma1, 3);
	bisa.split(ma1, 7);
	bisa.split(ma2, 2);
	mb1[0] = true; mb1[1] = true; mb2[2] = true;
	bisb.split(mb1, 3);
	bisb.split(mb1, 7);
	bisb.split(mb2, 2);
	tensor<3, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<3, double, allocator_t> bta(bisa), btb(bisb);

	permutation<3> perm210, perm021;
	perm210.permute(0, 1).permute(1, 2).permute(0, 1); // kji->ijk
	perm021.permute(1, 2); // kji->kij

	se_perm<3, double> cycle1(perm021, true);
	block_tensor_ctrl<3, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);

	//	Fill the input with random data

	btod_random<3>().perform(bta);
	bta.set_immutable();

	//	Make a copy

	btod_copy<3>(bta, perm210, 0.3).perform(btb);

	//	Create the reference

	tod_btconv<3>(bta).perform(ta);
	tod_copy<3>(ta, perm210, 0.3).perform(tb_ref);

	//	Compare against the reference

	tod_btconv<3>(btb).perform(tb);
	compare_ref<3>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} + a_{ij} \f$, no symmetry, no blocks
 **/
void btod_copy_test::test_add_nosym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nosym_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	//	Fill the input with random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta).perform(btb, 1.0);

	//	Compare against the reference

	tod_copy<2>(ta).perform(tb_ref, 1.0);
	tod_btconv<2>(btb).perform(tb);

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void btod_copy_test::test_2() throw(libtest::test_exception) {

	//
	//	b_ijkl = b_ijkl + 2.0 * a_ijkl
	//	Dimensions [ij]=10, [kl]=12, permutational symmetry
	//	Sym(B) = Sym(A)
	//

	static const char *testname = "btod_copy_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	dimensions<4> dimsb(dimsa);
	block_index_space<4> bisa(dimsa), bisb(dimsb);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 3);
	bisb.split(msk1, 5);
	bisb.split(msk2, 4);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1, perm2;
	perm1.permute(0, 1);
	perm2.permute(0, 1);
	se_perm<4, double> cycle1(perm1, true), cycle2(perm2, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	bta.set_immutable();

	//	Convert input block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, 2.0).perform(btb, 1.0);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 2.0);

	//	Compare against reference

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_copy_test::test_3() throw(libtest::test_exception) {

	//
	//	b_ikjl = a_ijkl
	//	Dimensions [ij]=10, [kl]=12, permutational symmetry
	//	Sym(B) = Sym(A)
	//

	static const char *testname = "btod_copy_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	block_index_space<4> bisa(dimsa);

	mask<4> msk1, msk2, msk3, msk4;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	msk3[0] = true; msk3[2] = true;
	msk4[1] = true; msk4[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	permutation<4> perm; perm.permute(1, 2);
	dimensions<4> dimsb(dimsa);
	block_index_space<4> bisb(bisa);
	dimsb.permute(perm);
	bisb.permute(perm);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1, perm2, perm3, perm4;
	perm1.permute(0, 1);
	perm2.permute(2, 3);
	perm3.permute(0, 2);
	perm4.permute(1, 3);
	se_perm<4, double> cycle1(perm1, true), cycle2(perm2, true);
	se_perm<4, double> cycle3(perm3, true), cycle4(perm4, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
//	ctrlb.req_sym_add_element(cycle3);
//	ctrlb.req_sym_add_element(cycle4);

	//	Load random data for input

	btod_random<4> rand;
	rand.perform(bta);
	rand.perform(btb);
	bta.set_immutable();

	//	Convert input block tensors to regular tensors

	tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, perm).perform(btb);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta, perm).perform(tb_ref);

	//	Compare against reference

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_copy_test::test_4() throw(libtest::test_exception) {

	//
	//	Copy to an empty block tensor with a coefficient
	//
	//	btod_copy<2> cp(A);
	//	cp.perform(B, 2.0);
	//

	static const char *testname = "btod_copy_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

	//	Fill in with random data

	btod_random<2>().perform(bta);
	bta.set_immutable();

	//	Make a copy and a reference

	btod_copy<2>(bta).perform(btb, 2.0);
	tod_btconv<2>(btb).perform(tb);
	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, 2.0).perform(tb_ref);

	//	Compare against the reference

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void btod_copy_test::test_dir_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_dir_1()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);
	block_tensor_ctrl<2, double> bta_ctrl(bta);

	//	Fill in with random data

	btod_random<2>().perform(bta);

	//	Make a copy

	btod_copy<2> cp(bta);
	cp.perform(btb_ref);
	cp.perform(btb, index<2>());

	//	Compare against the reference

	compare_ref<2>::compare(testname, btb, btb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void btod_copy_test::test_dir_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_dir_2()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);
	block_tensor_ctrl<2, double> bta_ctrl(bta), btb_ctrl(btb),
		btb_ref_ctrl(btb);

	permutation<2> perm1;
	perm1.permute(0, 1);
	se_perm<2, double> cycle(perm1, true);
	bta_ctrl.req_symmetry().insert(cycle);
	btb_ctrl.req_symmetry().insert(cycle);
	btb_ref_ctrl.req_symmetry().insert(cycle);

	//	Fill in with random data

	btod_random<2>().perform(bta);

	//	Make a copy

	permutation<2> perm;
	perm.permute(0, 1);
	btod_copy<2> cp(bta, perm, 2.0);
	cp.perform(btb_ref);
	cp.perform(btb, index<2>());

	//	Compare against the reference

	compare_ref<2>::compare(testname, btb, btb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void btod_copy_test::test_dir_3() throw(libtest::test_exception) {

	//
	//	b_ijkl = 2.0 * a_ijkl
	//	Dimensions [ij]=10, [kl]=12, permutational symmetry
	//	Sym(B) = Sym(A)
	//

	static const char *testname = "btod_copy_test::test_dir_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	dimensions<4> dimsb(dimsa);
	block_index_space<4> bisa(dimsa), bisb(dimsb);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 3);
	bisb.split(msk1, 5);
	bisb.split(msk2, 4);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb),
		btb_ref(bisb);

	//	Set up symmetry

	permutation<4> perm1, perm2;
	perm1.permute(0, 1);
	perm2.permute(2, 3);
	se_perm<4, double> cycle1(perm1, true), cycle2(perm2, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlb_ref(btb_ref);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);
	ctrlb_ref.req_symmetry().insert(cycle1);
	ctrlb_ref.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta);
	bta.set_immutable();

	//	Run the operation

	btod_copy<4> cp(bta, 2.0);
	cp.perform(btb_ref);
	orbit_list<4, double> orblst(ctrlb.req_symmetry());
	orbit_list<4, double>::iterator iorb = orblst.begin();
	for(; iorb != orblst.end(); iorb++) {
		cp.perform(btb, orblst.get_index(iorb));
	}

	//	Compare against reference

	compare_ref<4>::compare(testname, btb, btb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_copy_test::test_dir_4() throw(libtest::test_exception) {

	//
	//	b_ijkl = 2.0 * a_ijkl
	//	Dimensions [ij]=10, [kl]=12, permutational symmetry
	//	Sym(B) = Sym(A)
	//	One non-zero block
	//

	static const char *testname = "btod_copy_test::test_dir_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dimsa(index_range<4>(i1, i2));
	dimensions<4> dimsb(dimsa);
	block_index_space<4> bisa(dimsa), bisb(dimsb);

	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;

	bisa.split(msk1, 3);
	bisa.split(msk1, 5);
	bisa.split(msk2, 4);

	bisb.split(msk1, 3);
	bisb.split(msk1, 5);
	bisb.split(msk2, 4);

	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb),
		btb_ref(bisb);

	index<4> idx;
	idx[0] = 1; idx[1] = 2; idx[0] = 0; idx[1] = 1;

	//	Set up symmetry

	permutation<4> perm1, perm2;
	perm1.permute(0, 1);
	perm2.permute(2, 3);
	se_perm<4, double> cycle1(perm1, true), cycle2(perm2, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlb_ref(btb_ref);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);
	ctrlb_ref.req_symmetry().insert(cycle1);
	ctrlb_ref.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta, idx);
	btod_random<4>().perform(btb);
	bta.set_immutable();

	//	Run the operation

	btod_copy<4> cp(bta, 2.0);
	cp.perform(btb_ref);
	orbit_list<4, double> orblst(ctrlb.req_symmetry());
	orbit_list<4, double>::iterator iorb = orblst.begin();
	for(; iorb != orblst.end(); iorb++) {
		cp.perform(btb, orblst.get_index(iorb));
	}

	//	Compare against reference

	compare_ref<4>::compare(testname, btb, btb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
