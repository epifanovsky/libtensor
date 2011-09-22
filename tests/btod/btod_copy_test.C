#include <libtensor/core/allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include "../compare_ref.h"
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
	test_sym_3();
	test_sym_4();

	test_add_nosym_1();
	test_add_nosym_2();
	test_add_nosym_3();
	test_add_nosym_4();
	test_add_eqsym_1();
	test_add_eqsym_2();
	test_add_eqsym_3();
	test_add_eqsym_4();
	test_add_eqsym_5();
	test_add_nesym_1();
	test_add_nesym_2();
	test_add_nesym_3();
	test_add_nesym_4();
	test_add_nesym_5();
	test_add_nesym_5_sp();
	test_add_nesym_6();
	test_add_nesym_7_sp1();
	test_add_nesym_7_sp2();
	test_add_nesym_7_sp3();

	//~ test_dir_1();
	//~ test_dir_2();
	//~ test_dir_3();
	//~ test_dir_4();
}


/**	\test \f$ b_{ij} = a_{ij} \f$, zero tensor
 **/
void btod_copy_test::test_zero_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_zero_1()";

	typedef std_allocator<double> allocator_t;

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
		abs_index<2> blkidx(orb.get_abs_canonical_index(), bidims);
		if(!btb_ctrl.req_is_zero_block(blkidx.get_index())) {
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

	typedef std_allocator<double> allocator_t;

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
        abs_index<2> blkidx(orb.get_abs_canonical_index(), bidims);
		if(!btb_ctrl.req_is_zero_block(blkidx.get_index())) {
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

	typedef std_allocator<double> allocator_t;

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

	typedef std_allocator<double> allocator_t;

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

	typedef std_allocator<double> allocator_t;

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

	typedef std_allocator<double> allocator_t;

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
	btod_random<2>().perform(btb);
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

	typedef std_allocator<double> allocator_t;

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

	typedef std_allocator<double> allocator_t;

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

	typedef std_allocator<double> allocator_t;

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


/**	\test \f$ b_{ijkl} = -a_{ijkl} \f$, perm symmetry,
		dim(ij)=10, dim(kl)=12, blocks, non-zero initial B
 **/
void btod_copy_test::test_sym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_sym_4()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1, m2;
	m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisa.split(m2, 4);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	bisb.split(m2, 4);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1023, perm0132;
	perm1023.permute(0, 1);
	perm0132.permute(2, 3);
	se_perm<4, double> cycle1(perm1023, true), cycle2(perm0132, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, -1.0).perform(btb);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta, -1.0).perform(tb_ref);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} + a_{ij} \f$, no symmetry, no blocks
 **/
void btod_copy_test::test_add_nosym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nosym_1()";

	typedef std_allocator<double> allocator_t;

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


/**	\test \f$ b_{ij} = b_{ij} + 2 a_{ji} \f$, no symmetry,
		3 blocks along each direction
 **/
void btod_copy_test::test_add_nosym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nosym_2()";

	typedef std_allocator<double> allocator_t;

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
	btod_random<2>().perform(btb);
	bta.set_immutable();
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta, perm10).perform(btb, 2.0);

	//	Create the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, perm10, 2.0).perform(tb_ref, 1.0);

	//	Compare against the reference

	tod_btconv<2>(btb).perform(tb);
	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 2 a_{ijkl} \f$, no symmetry,
		dim(ij)=10, dim(kl)=12, blocks
 **/
void btod_copy_test::test_add_nosym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nosym_3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1, m2;
	m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisa.split(m2, 4);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	bisb.split(m2, 4);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, 2.0).perform(btb, 1.0);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta, 2.0).perform(tb_ref, 1.0);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} + 2 a_{ji} \f$, no symmetry,
		3 blocks along each direction, empty initial B
 **/
void btod_copy_test::test_add_nosym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nosym_4()";

	typedef std_allocator<double> allocator_t;

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
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta, perm10).perform(btb, 2.0);

	//	Create the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, perm10, 2.0).perform(tb_ref, 1.0);

	//	Compare against the reference

	tod_btconv<2>(btb).perform(tb);
	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} + a_{ij} \f$, equal perm symmetry,
		3 blocks along each direction
 **/
void btod_copy_test::test_add_eqsym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_eqsym_1()";

	typedef std_allocator<double> allocator_t;

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
	se_perm<2, double> cycle1(perm10, true);
	block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle1);
	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta).perform(btb, 1.0);
	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta).perform(tb_ref, 1.0);

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} - a_{ji} \f$, equal perm antisymmetry,
		3 blocks along each direction
 **/
void btod_copy_test::test_add_eqsym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_eqsym_2()";

	typedef std_allocator<double> allocator_t;

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

	se_perm<2, double> cycle1(perm10, false);
	block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle1);
	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta, perm10).perform(btb, -1.0);
	tod_btconv<2>(btb).perform(tb);

	//	Create the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta, perm10).perform(tb_ref, -1.0);

	//	Compare against the reference

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ijk} = b_{ijk} + 0.75 a_{kji} \f$, equal perm symmetry,
		3 blocks along each direction
 **/
void btod_copy_test::test_add_eqsym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_eqsym_3()";

	typedef std_allocator<double> allocator_t;

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

	permutation<3> perm210, perm021, perm102;
	perm210.permute(0, 1).permute(1, 2).permute(0, 1); // kji->ijk
	perm021.permute(1, 2); // kji->kij
	perm102.permute(0, 1); // ijk->jik

	se_perm<3, double> cycle1(perm021, true);
	se_perm<3, double> cycle2(perm102, true);
	block_tensor_ctrl<3, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Fill the input with random data

	btod_random<3>().perform(bta);
	btod_random<3>().perform(btb);
	bta.set_immutable();
	tod_btconv<3>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<3>(bta, perm210).perform(btb, 0.75);
	tod_btconv<3>(btb).perform(tb);

	//	Create the reference

	tod_btconv<3>(bta).perform(ta);
	tod_copy<3>(ta, perm210, 1.5).perform(tb_ref, 0.5);

	//	Compare against the reference

	compare_ref<3>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 2 a_{ijkl} \f$, equal perm symmetry,
		dim(ij)=10, dim(kl)=12, blocks
 **/
void btod_copy_test::test_add_eqsym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_eqsym_4()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1, m2;
	m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisa.split(m2, 4);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	bisb.split(m2, 4);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1023, perm0132;
	perm1023.permute(0, 1);
	perm0132.permute(2, 3);
	se_perm<4, double> cycle1(perm1023, true), cycle2(perm0132, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, 2.0).perform(btb, 1.0);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 2.0);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ikjl} = b_{ikjl} + 0.5 a_{ijkl} \f$,
		equal perm antisymmetry, dim(ij)=10, dim(kl)=12, blocks
 **/
void btod_copy_test::test_add_eqsym_5() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_eqsym_5()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1a, i2a, i1b, i2b;
	i2a[0] = 9; i2a[1] = 9; i2a[2] = 11; i2a[3] = 11;
	i2b[0] = 9; i2b[1] = 11; i2b[2] = 9; i2b[3] = 11;
	dimensions<4> dima(index_range<4>(i1a, i2a));
	dimensions<4> dimb(index_range<4>(i1b, i2b));
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1a, m2a, m1b, m2b;
	m1a[0] = true; m1a[1] = true; m2a[2] = true; m2a[3] = true;
	m1b[0] = true; m2b[1] = true; m1b[2] = true; m2b[3] = true;
	bisa.split(m1a, 3);
	bisa.split(m1a, 5);
	bisa.split(m2a, 4);
	bisb.split(m1b, 3);
	bisb.split(m1b, 5);
	bisb.split(m2b, 4);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);
	permutation<4> perm0213;
	perm0213.permute(1, 2);

	//	Set up symmetry

	permutation<4> perm1032, perm2301;
	perm1032.permute(0, 1).permute(2, 3);
	perm2301.permute(0, 2).permute(1, 3);
	se_perm<4, double> cycle1(perm1032, false), cycle2(perm2301, false);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, perm0213, 1.0).perform(btb, 0.5);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta, perm0213, 0.5).perform(tb_ref, 1.0);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} + a_{ij} \f$, unequal perm symmetry,
		Sym(A) > Sym(B) = Sym(B') = Sym(0), blocks
 **/
void btod_copy_test::test_add_nesym_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_1()";

	typedef std_allocator<double> allocator_t;

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
	se_perm<2, double> cycle1(perm10, true);
	block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta).perform(btb, 1.0);
	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta).perform(tb_ref, 1.0);

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ij} = b_{ij} + a_{ij} \f$, unequal perm symmetry,
		Sym(B) > Sym(A) = Sym(B') = Sym(0), blocks
 **/
void btod_copy_test::test_add_nesym_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_2()";

	typedef std_allocator<double> allocator_t;

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
	se_perm<2, double> cycle1(perm10, true);
	block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
	ctrlb.req_symmetry().insert(cycle1);
	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	bta.set_immutable();
	tod_btconv<2>(btb).perform(tb_ref);

	//	Make a copy

	btod_copy<2>(bta).perform(btb, 1.0);
	tod_btconv<2>(btb).perform(tb);

	//	Compare against the reference

	tod_btconv<2>(bta).perform(ta);
	tod_copy<2>(ta).perform(tb_ref, 1.0);

	compare_ref<2>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 1.5 a_{ijkl} \f$, unequal perm symmetry,
		Sym(A) > Sym(B) = Sym(B') != Sym(0), blocks
 **/
void btod_copy_test::test_add_nesym_3() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1230, perm1023, perm1032;
	perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
	perm1023.permute(0, 1);
	perm1032.permute(0, 1).permute(2, 3);
	se_perm<4, double> cycle1(perm1230, true), cycle2(perm1023, true),
		cycle3(perm1032, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle3);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, 1.5);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 1.5);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 1.5 a_{ijkl} \f$, unequal perm symmetry,
		Sym(B) > Sym(A) = Sym(B') != Sym(0), blocks
 **/
void btod_copy_test::test_add_nesym_4() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_4()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1230, perm1023, perm1032;
	perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
	perm1023.permute(0, 1);
	perm1032.permute(0, 1).permute(2, 3);
	se_perm<4, double> cycle1(perm1230, true), cycle2(perm1023, true),
		cycle3(perm1032, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle3);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, 1.5);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 1.5);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 1.5 a_{ijkl} \f$, unequal perm symmetry,
		blocks
 **/
void btod_copy_test::test_add_nesym_5() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_5()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1230, perm1023, perm1032;
	perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
	perm1023.permute(0, 1);
	perm1032.permute(0, 1).permute(2, 3);
	se_perm<4, double> cycle1(perm1230, true), cycle2(perm1023, true),
		cycle3(perm1032, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle3);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, 1.5);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 1.5);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 1.5 a_{ijkl} \f$, unequal perm symmetry,
		sparse block structure
 **/
void btod_copy_test::test_add_nesym_5_sp() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_5_sp()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1230, perm1023, perm1032;
	perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
	perm1023.permute(0, 1);
	perm1032.permute(0, 1).permute(2, 3);
	se_perm<4, double> cycle1(perm1230, true), cycle2(perm1023, true),
		cycle3(perm1032, true);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle3);
	ctrlb.req_symmetry().insert(cycle1);
	ctrlb.req_symmetry().insert(cycle2);

	//	Load random data for input

	index<4> i0000, i0001, i0011, i0012, i0101, i0112, i1111, i2222;
	i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1;
	i0012[2] = 1; i0012[3] = 2;
	i0101[1] = 1; i0101[3] = 1;
	i0112[1] = 1; i0112[2] = 1; i0112[3] = 2;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	i2222[0] = 2; i2222[1] = 2; i2222[2] = 2; i2222[3] = 2;
	btod_random<4>().perform(bta, i0101);
	btod_random<4>().perform(bta, i0112);
	btod_random<4>().perform(bta, i1111);
	btod_random<4>().perform(bta, i2222);
	btod_random<4>().perform(btb, i0000);
	btod_random<4>().perform(btb, i0001);
	btod_random<4>().perform(btb, i0011);
	btod_random<4>().perform(btb, i0012);
	btod_random<4>().perform(btb, i1111);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, 1.5);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 1.5);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{lkji} = b_{lkji} - 0.1 a_{ijkl} \f$, unequal mixed perm
		symmetry and antisymmetry, blocks
 **/
void btod_copy_test::test_add_nesym_6() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_6()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	permutation<4> perm1230, perm1023, perm1032, perm3210;
	perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
	perm1023.permute(0, 1);
	perm1032.permute(0, 1).permute(2, 3);
	perm3210.permute(0, 1).permute(1, 2).permute(2, 3).permute(0, 2);
	se_perm<4, double> cycle1(perm1230, true), cycle2(perm1023, true),
		cycle3(perm1032, false);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(cycle1);
	ctrla.req_symmetry().insert(cycle2);
	ctrlb.req_symmetry().insert(cycle3);

	//	Load random data for input

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta, perm3210).perform(btb, -0.1);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta, perm3210).perform(tb_ref, -0.1);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} - 2.0 a_{ijkl} \f$, unequal perm symmetry,
		sparse block structure. Sym(A)=S3*C1, Sym(B)=S2*S2,
		Sym(A+B)=S2*C1*C1.
		Check addition
		C[0,1,0,2] = A[0,0,1,2] + B[0,1,0,2],
		C[0,1,2,0] = A[0,1,2,0] + B[0,1,0,2],
		A[0,0,1,2] = 0
 **/
void btod_copy_test::test_add_nesym_7_sp1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_7_sp1()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1).permute(1, 2), true));
	ctrla.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	ctrlb.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	ctrlb.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(2, 3), true));

	//	Load random data for input

	index<4> i0012, i0102, i0120;
	i0012[2] = 1; i0012[3] = 2;
	i0102[1] = 1; i0102[3] = 2;
	i0120[1] = 1; i0120[2] = 2;
	btod_random<4>().perform(bta, i0120);
	btod_random<4>().perform(btb, i0102);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, -2.0);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, -2.0);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + a_{ijkl} \f$, unequal perm symmetry,
		sparse block structure. Sym(A)=S3*C1, Sym(B)=S2*S2,
		Sym(A+B)=S2*C1*C1.
		Check addition
		C[0,1,0,2] = B[0,1,0,2],
		C[0,1,2,0] = B[0,1,0,2],
 **/
void btod_copy_test::test_add_nesym_7_sp2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_7_sp2()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1).permute(1, 2), true));
	ctrla.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	ctrlb.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	ctrlb.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(2, 3), true));

	//	Load random data for input

	index<4> i0012, i0102, i0120;
	i0012[2] = 1; i0012[3] = 2;
	i0102[1] = 1; i0102[3] = 2;
	i0120[1] = 1; i0120[2] = 2;
	btod_random<4>().perform(btb, i0102);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, -2.0);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, -2.0);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test \f$ b_{ijkl} = b_{ijkl} + 2.0 a_{ijkl} \f$, unequal perm symmetry,
		sparse block structure. Sym(A)=S2*S2, Sym(B)=S3*C1,
		Sym(A+B)=S2*C1*C1.
		Check addition
		C[0,1,0,2] = A[0,1,0,2] + B[0,0,1,2],
		C[0,1,2,0] = A[0,1,0,2] + B[0,1,2,0],
		B[0,0,1,2] = 0
 **/
void btod_copy_test::test_add_nesym_7_sp3() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_add_nesym_7_sp3()";

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 11; i2[1] = 11; i2[2] = 11; i2[3] = 11;
	dimensions<4> dima(index_range<4>(i1, i2));
	dimensions<4> dimb(dima);
	block_index_space<4> bisa(dima), bisb(dimb);
	mask<4> m1;
	m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
	bisa.split(m1, 3);
	bisa.split(m1, 5);
	bisb.split(m1, 3);
	bisb.split(m1, 5);
	tensor<4, double, allocator_t> ta(dima), tb(dimb), tb_ref(dimb);
	block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);

	//	Set up symmetry

	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
	ctrla.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));
	ctrla.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(2, 3), true));
	ctrlb.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1).permute(1, 2), true));
	ctrlb.req_symmetry().insert(se_perm<4, double>(
		permutation<4>().permute(0, 1), true));

	//	Load random data for input

	index<4> i0012, i0102, i0120;
	i0012[2] = 1; i0012[3] = 2;
	i0102[1] = 1; i0102[3] = 2;
	i0120[1] = 1; i0120[2] = 2;
	btod_random<4>().perform(bta, i0102);
	btod_random<4>().perform(btb, i0120);
	bta.set_immutable();
	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb_ref);

	//	Run the operation

	btod_copy<4>(bta).perform(btb, 2.0);
	tod_btconv<4>(btb).perform(tb);

	//	Compute the reference

	tod_copy<4>(ta).perform(tb_ref, 2.0);

	//	Compare against the reference

	compare_ref<4>::compare(testname, tb, tb_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void btod_copy_test::test_dir_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_dir_1()";

	typedef std_allocator<double> allocator_t;
	typedef tensor<2, double, allocator_t> tensor_t;
	typedef tensor_ctrl<2, double> tensor_ctrl_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;
/*
	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	tensor<2, double, allocator_t> ta(dims), tb(dims);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
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
	}*/
}


void btod_copy_test::test_dir_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_dir_2()";
/*
	typedef std_allocator<double> allocator_t;
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
	}*/
}


void btod_copy_test::test_dir_3() throw(libtest::test_exception) {

	//
	//	b_ijkl = 2.0 * a_ijkl
	//	Dimensions [ij]=10, [kl]=12, permutational symmetry
	//	Sym(B) = Sym(A)
	//

	static const char *testname = "btod_copy_test::test_dir_3()";

	typedef std_allocator<double> allocator_t;
/*
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
	}*/
}


void btod_copy_test::test_dir_4() throw(libtest::test_exception) {

	//
	//	b_ijkl = 2.0 * a_ijkl
	//	Dimensions [ij]=10, [kl]=12, permutational symmetry
	//	Sym(B) = Sym(A)
	//	One non-zero block
	//

	static const char *testname = "btod_copy_test::test_dir_4()";

	typedef std_allocator<double> allocator_t;
/*
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
	}*/
}


} // namespace libtensor
