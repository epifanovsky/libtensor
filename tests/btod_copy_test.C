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
		orbit<2, double> orb(btb_ctrl.req_symmetry(), *iorbit);
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
	bis.split(0, 3);
	bis.split(0, 6);
	bis.split(1, 5);
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
		orbit<2, double> orb(btb_ctrl.req_symmetry(), *iorbit);
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

} // namespace libtensor
