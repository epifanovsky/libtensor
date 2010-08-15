#include <sstream>
#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/tensor.h>
#include <libtensor/btod/btod_import_raw.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_select.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_select.h>
#include "btod_select_test.h"


namespace libtensor {


void btod_select_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();

	point_group_table pg("x", 2);
	pg.add_product(0, 0, 0);
	pg.add_product(0, 1, 1);
	pg.add_product(1, 1, 0);

	product_table_container::get_instance().add(pg);

	try {

	test_5();

	}
	catch (...) {
		product_table_container::get_instance().erase("x");
		throw;
	}

	product_table_container::get_instance().erase("x");

}

/** \test Selecting 5 elements from random block tensor (1 block)
 **/
void btod_select_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	tod_select<2>::list_t tlist;
	btod_select<2>::list_t btlist;


	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	btod_select<2>(bt).perform(btlist, 5);
	tod_select<2>(t_ref).perform(tlist, 5);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	tod_select<2>::list_t::iterator it = tlist.begin();
	btod_select<2>::list_t::iterator ibt = btlist.begin();
	while (it != tlist.end() && ibt != btlist.end()) {
		if ( it->value != ibt->value ) {
			std::ostringstream oss;
			oss << "Value of list element does not match reference "
					<< "(found: " << ibt->value
					<< ", expected: " << it->value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		index<2> idx=bis.get_block_start(ibt->bidx);
		idx[0]+=ibt->idx[0];
		idx[1]+=ibt->idx[1];
		if (! idx.equals(it->idx)) {
			std::ostringstream oss;
			oss << "Index of list element does not match reference "
					<< "(found: " << idx
					<< ", expected: " << it->idx << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		it++; ibt++;
	}

}


/** \test Selecting 10 elements from random block tensor (multiple blocks)
 **/
void btod_select_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10;
	m01[1] = true; m10[0] = true;
	bis.split(m10, 3);
	bis.split(m01, 4);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	tod_select<2>::list_t tlist;
	btod_select<2>::list_t btlist;

	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	btod_select<2>(bt).perform(btlist, 10);
	tod_select<2>(t_ref).perform(tlist, 10);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	tod_select<2>::list_t::iterator it = tlist.begin();
	btod_select<2>::list_t::iterator ibt = btlist.begin();
	while (it != tlist.end() && ibt != btlist.end()) {
		if (it->value != ibt->value) {
			std::ostringstream oss;
			oss << "Value of list element does not match reference "
					<< "(found: " << ibt->value
					<< ", expected: " << it->value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}
		index<2> idx=bis.get_block_start(ibt->bidx);
		idx[0]+=ibt->idx[0];
		idx[1]+=ibt->idx[1];
		if (! idx.equals(it->idx)) {
			std::ostringstream oss;
			oss << "Index of list element does not match reference "
					<< "(found: " << idx
					<< ", expected: " << it->idx << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		it++; ibt++;
	}
}

/** \test Selecting 10 elements from random block tensor with symmetry
 **/
void btod_select_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11; m11[0] = true; m11[1] = true;
	bis.split(m11, 3);
	bis.split(m11, 5);
	block_tensor<2, double, allocator_t> bt(bis);
	block_tensor_ctrl<2, double> ctrl(bt);
	ctrl.req_symmetry().insert(
			se_perm<2, double>(permutation<2>().permute(0, 1), true));

	tensor<2, double, allocator_t> t_ref(dims);
	tod_select<2>::list_t tlist;
	btod_select<2>::list_t btlist;

	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	btod_select<2>(bt).perform(btlist, 10);
	tod_select<2>(t_ref).perform(tlist, 30);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	double last_value = 0.0;
	for (btod_select<2>::list_t::iterator ibt = btlist.begin();
			ibt != btlist.end(); ibt++) {

		tod_select<2>::list_t::iterator it = tlist.begin();
		while (it != tlist.end() && it->value != ibt->value) it++;

		while (it != tlist.end() && it->value == ibt->value) {

			index<2> idx = bis.get_block_start(ibt->bidx);
			idx[0]+=ibt->idx[0];
			idx[1]+=ibt->idx[1];
			if (idx.equals(it->idx)) break;

			it++;
		}

		if (it == tlist.end() || it->value != ibt->value) {
			std::ostringstream oss;
			oss << "List element not found in reference "
					<< "(" << ibt->bidx << ", "
					<< ibt->idx << ": " << ibt->value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		if (fabs(ibt->value) < last_value) {
			std::ostringstream oss;
			oss << "Invalid ordering of values "
					<< "(fabs(" << ibt->value << ") < "
					<< last_value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());

		}
		last_value = fabs(ibt->value);
	}
}

/** \test Selecting 10 elements from random block tensor without symmetry,
 	 but permutational symmetry added.
 **/
void btod_select_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_4()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 8; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11;
	m11[1] = true; m11[0] = true;
	bis.split(m11, 3);
	bis.split(m11, 5);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	symmetry<2, double> sym(bis);
	{
	se_perm<2, double> se(permutation<2>().permute(0, 1), true);
	sym.insert(se);
	block_tensor_ctrl<2, double> ctrl(bt_ref);
	ctrl.req_symmetry().insert(se);
	}

	btod_select<2>::list_t btlist, btlist_ref;

	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt_ref);

	tensor<2, double, allocator_t> tmp(dims);
	tod_btconv<2>(bt_ref).perform(tmp);
	tensor_ctrl<2, double> ctrl(tmp);
	const double *ptr = ctrl.req_const_dataptr();
	btod_import_raw<2>(ptr, dims).perform(bt);
	ctrl.ret_dataptr(ptr);

	btod_select<2>(bt, sym).perform(btlist, 10);
	btod_select<2>(bt_ref).perform(btlist_ref, 10);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	for (btod_select<2>::list_t::iterator ibt = btlist.begin(),
			ibt_ref = btlist_ref.begin();
			ibt != btlist.end(); ibt++, ibt_ref++) {

		if (ibt->value != ibt_ref->value ||
				! ibt->bidx.equals(ibt_ref->bidx) ||
				! ibt->idx.equals(ibt_ref->idx)) {

			std::ostringstream oss;
			oss << "List element does not match reference "
					<< "(found: " << ibt->bidx << ", " << ibt->idx
					<< ": " << ibt->value << ", expected: " << ibt_ref->bidx
					<< ", " << ibt_ref->idx << ": " << ibt->value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}
	}
}

/** \test Selecting 10 elements from random block tensor without symmetry,
 	 but irrep symmetry added.
 **/
void btod_select_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_5()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10;
	m01[1] = true; m10[0] = true;
	bis.split(m10, 3);
	bis.split(m01, 4);
	block_tensor<2, double, allocator_t> bt(bis);
	symmetry<2, double> sym(bis);
	{

	se_label<2, double> se(bis.get_block_index_dims(), "x");
	mask<2> m; m[0] = true; m[1] = true;
	se.assign(m, 0, 0);
	se.assign(m, 1, 1);
	se.add_target(1);
	sym.insert(se);
	}

	btod_select<2>::list_t btlist;

	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt);

	btod_select<2>(bt, sym).perform(btlist, 10);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	double last_value = 0.0;
	index<2> vidx1, vidx2;
	vidx1[0] = 0; vidx1[1] = 1;
	vidx2[0] = 1; vidx2[1] = 0;

	for (btod_select<2>::list_t::iterator ibt = btlist.begin();
			ibt != btlist.end(); ibt++) {

		if (! ibt->bidx.equals(vidx1) && ! ibt->bidx.equals(vidx2)) {
			std::ostringstream oss;
			oss << "Block index of list element not valid "
					<< "(found: " << ibt->bidx << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		if (fabs(ibt->value) < last_value) {
			std::ostringstream oss;
			oss << "Invalid ordering of values "
					<< "(fabs(" << ibt->value << ") < "
					<< last_value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());

		}
		last_value = fabs(ibt->value);
	}
}

} // namespace libtensor
