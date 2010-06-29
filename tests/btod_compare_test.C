#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_compare.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include "btod_compare_test.h"

namespace libtensor {


void btod_compare_test::perform() throw(libtest::test_exception) {

	test_exc();
	test_operation();
}


void btod_compare_test::test_exc() throw(libtest::test_exception) {
	typedef index<2> index_t;
	typedef index_range<2> index_range_t;
	typedef dimensions<2> dimensions_t;
	typedef mask<2> mask_t;
	typedef block_index_space<2> block_index_space_t;
	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;


	index_t i1, i2, i3;
	i2[0]=5; i2[1]=5;
	i3[0]=7; i3[1]=7;
	index_range_t ir1(i1,i2), ir2(i1,i3);
	dimensions_t dim1(ir1), dim2(ir2);
	block_index_space_t bis1(dim1);
	mask_t mask;
	mask[0]=true; mask[1]=true;
	bis1.split(mask,3);
	block_tensor_t bt1(bis1);


	bool ok = false;
	try {
		block_index_space_t bis2(dim2);
		bis2.split(mask,3);
		block_tensor_t bt2(bis2);
		btod_compare<2> btc(bt1, bt2, 0);
	} catch(exception e) {
		ok = true;
	}

	if(!ok) {
		fail_test("tod_compare_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}

	ok = false;
	try {
		block_index_space_t bis2(dim2);
		mask[1]=false;
		bis2.split(mask,4);
		mask[0]=false; mask[1]=true;
		bis2.split(mask,2);
		block_tensor_t bt2(bis2);

		btod_compare<2> btc(bt1, bt2, 0);
	} catch(exception e) {
		ok = true;
	}

	if(!ok) {
		fail_test("tod_compare_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}


}


void btod_compare_test::test_operation() throw(libtest::test_exception) {

	static const char *testname = "btod_compare_test::test_operation()";

	typedef index<2> index_t;
	typedef index_range<2> index_range_t;
	typedef dimensions<2> dimensions_t;
	typedef mask<2> mask_t;
	typedef block_index_space<2> block_index_space_t;
	typedef libvmm::std_allocator<double> allocator_t;
	typedef block_tensor<2, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;


	index_t i1, i2;
	i2[0]=5; i2[1]=5;
	index_range_t ir(i1,i2);
	dimensions_t dim(ir);
	block_index_space_t bis(dim);
	mask_t mask;
	mask[0]=true; mask[1]=true;
	bis.split(mask,3);
	block_tensor_t bt1(bis), bt2(bis);

	btod_random<2> randr;
	randr.perform(bt1);

	btod_copy<2> docopy(bt1);
	docopy.perform(bt2);

	index_t block_idx, inblock_idx;
	block_idx[0]=1; block_idx[1]=0;
	inblock_idx[0]=1; inblock_idx[1]=1;

	block_tensor_ctrl_t btctrl(bt2);
	tensor_i<2,double>& t2=btctrl.req_block(block_idx);
	double diff1, diff2;
	{
		tensor_ctrl<2,double> tctrl(t2);
		double *ptr=tctrl.req_dataptr();
		diff1=ptr[4];
		ptr[4]-=1.0;
		diff2=ptr[4];
		tctrl.ret_dataptr(ptr);
	}
	btctrl.ret_block(block_idx);
	bt1.set_immutable();
	bt2.set_immutable();

	btod_compare<2> cmp(bt1, bt2, 1e-7);

	if(cmp.compare()) {
		fail_test(testname, __FILE__, __LINE__,
			"Operation failed to find the difference.");
	}

//	if( ! op1.get_diff().m_number_of_orbits ) {
//		fail_test(testname, __FILE__, __LINE__,
//			"btod_compare returned different number of orbits");
//	}
//	if( ! op1.get_diff().m_similar_orbit ) {
//		fail_test(testname, __FILE__, __LINE__,
//			"btod_compare returned different orbit");
//	}
//	if( ! op1.get_diff().m_canonical_block_index_1
//			.equals(op1.get_diff().m_canonical_block_index_2) ) {
//		fail_test(testname, __FILE__, __LINE__,
//			"btod_compare returned different canonical blocks");
//	}
//	if( ! inblock_idx.equals(op1.get_diff().m_inblock) ) {
//		fail_test(testname, __FILE__, __LINE__,
//			"btod_compare returned an incorrect index");
//	}
//	if( op1.get_diff().m_diff_elem_1 != diff1 ||
//		op1.get_diff().m_diff_elem_2 != diff2) {
//		fail_test(testname, __FILE__,
//			__LINE__, "btod_compare returned an incorrect "
//			"element value");
//	}

}


} // namespace libtensor
