#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libvmm.h>
#include <libtensor.h>
#include "compare_ref.h"
#include "btod_copy_test.h"

namespace libtensor {


void btod_copy_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	test_zero_1();
	test_1();
	test_2();
	test_dir_1();
	test_dir_2();
	test_dir_3();
	test_dir_4();
}


void btod_copy_test::test_zero_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_zero_1()";

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
	dimensions<2> bidims(bis.get_block_index_dims());
	tensor_t ta(dims), tb(dims);
	block_tensor_t bta(bis), btb(bis);
	block_tensor_ctrl_t btb_ctrl(btb);

	// Fill in the output with random data

	index<2> i_00;
	tensor_i<2, double> &blk_00 = btb_ctrl.req_block(i_00);
	tensor_ctrl_t blk_00_ctrl(blk_00);
	double *ptr = blk_00_ctrl.req_dataptr();
	size_t sz = blk_00.get_dims().get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr[i] = drand48();
	}
	blk_00_ctrl.ret_dataptr(ptr); ptr = NULL;
	btb_ctrl.ret_block(i_00);

	// Make a copy

	btod_copy<2> cp(bta);
	cp.perform(btb);

	// The set of non-zero blocks in the output must be empty now

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


void btod_copy_test::test_zero_2() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_zero_2()";

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
	mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
	bis.split(msk1, 3);
	bis.split(msk1, 6);
	bis.split(msk2, 5);
	dimensions<2> bidims(bis.get_block_index_dims());
	tensor_t ta(dims), tb(dims);
	block_tensor_t bta(bis), btb(bis);
	block_tensor_ctrl_t btb_ctrl(btb);

	// Fill in the output with random data

	dimensions<2> blk_dims = bis.get_block_index_dims();
	index<2> iblk;
	do {
		tensor_i<2, double> &blk = btb_ctrl.req_block(iblk);
		tensor_ctrl_t blk_ctrl(blk);
		double *ptr = blk_ctrl.req_dataptr();
		size_t sz = blk.get_dims().get_size();
		for(size_t i = 0; i < sz; i++) {
			ptr[i] = drand48();
		}
		blk_ctrl.ret_dataptr(ptr); ptr = NULL;
		btb_ctrl.ret_block(iblk);
	} while(blk_dims.inc_index(iblk));

	// Make a copy

	btod_copy<2> cp(bta);
	cp.perform(btb);

	// The set of non-zero blocks in the output must be empty now

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


void btod_copy_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_copy_test::test_1()";

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
	tensor_t ta(dims), tb(dims);
	block_tensor_t bta(bis), btb(bis);
	block_tensor_ctrl_t bta_ctrl(bta);

	// Fill in with random data

	index<2> i_00;
	tensor_i<2, double> &blk_00 = bta_ctrl.req_block(i_00);
	tensor_ctrl_t blk_00_ctrl(blk_00);
	double *ptr = blk_00_ctrl.req_dataptr();
	size_t sz = blk_00.get_dims().get_size();
	for(size_t i = 0; i < sz; i++) {
		ptr[i] = drand48();
	}
	blk_00_ctrl.ret_dataptr(ptr); ptr = NULL;
	bta_ctrl.ret_block(i_00);

	// Make a copy

	btod_copy<2> cp(bta);
	cp.perform(btb);

	// Compare against the reference

	tod_btconv<2> conva(bta), convb(btb);
	conva.perform(ta);
	convb.perform(tb);
	compare_ref<2>::compare(testname, tb, ta, 0.0);

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

	mask<2> msk;
	msk[0] = true; msk[1] = true;
	symel_cycleperm<2, double> cycle(2, msk);
	bta_ctrl.req_sym_add_element(cycle);
	btb_ctrl.req_sym_add_element(cycle);
	btb_ref_ctrl.req_sym_add_element(cycle);

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

	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlb_ref(btb_ref);
	ctrla.req_sym_add_element(cycle1);
	ctrla.req_sym_add_element(cycle2);
	ctrlb.req_sym_add_element(cycle1);
	ctrlb.req_sym_add_element(cycle2);
	ctrlb_ref.req_sym_add_element(cycle1);
	ctrlb_ref.req_sym_add_element(cycle2);

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

	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlb_ref(btb_ref);
	ctrla.req_sym_add_element(cycle1);
	ctrla.req_sym_add_element(cycle2);
	ctrlb.req_sym_add_element(cycle1);
	ctrlb.req_sym_add_element(cycle2);
	ctrlb_ref.req_sym_add_element(cycle1);
	ctrlb_ref.req_sym_add_element(cycle2);

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
