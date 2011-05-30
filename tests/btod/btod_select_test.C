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
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_select.h>
#include "btod_select_test.h"


namespace libtensor {


void btod_select_test::perform() throw(libtest::test_exception) {

	point_group_table pg("x", 2);
	pg.add_product(0, 0, 0);
	pg.add_product(0, 1, 1);
	pg.add_product(1, 1, 0);

	product_table_container::get_instance().add(pg);

	try {

	test_1<compare4absmax>(2);
	test_1<compare4absmin>(4);
	test_1<compare4max>(7);
	test_1<compare4min>(9);

	test_2<compare4absmax>(3);
	test_2<compare4absmin>(9);
	test_2<compare4max>(27);
	test_2<compare4min>(45);

	test_3<compare4absmax>(4, true);
	test_3<compare4absmin>(8, true);
	test_3<compare4max>(16, true);
	test_3<compare4min>(32, true);

	test_3<compare4absmax>(3, false);
	test_3<compare4absmin>(6, false);
	test_3<compare4max>(13, false);
	test_3<compare4min>(45, false);

	test_4<compare4absmax>(5, true);
	test_4<compare4absmin>(10, true);
	test_4<compare4max>(15, true);
	test_4<compare4min>(30, true);

	test_4<compare4absmax>(7, false);
	test_4<compare4absmin>(17, false);
	test_4<compare4max>(25, false);
	test_4<compare4min>(33, false);

	test_5<compare4absmax>(7);
	test_5<compare4absmin>(12);
	test_5<compare4max>(19);
	test_5<compare4min>(3);

	test_6<compare4absmax>(20);
	test_6<compare4absmin>(30);
	test_6<compare4max>(15);
	test_6<compare4min>(8);

	}
	catch (...) {
		product_table_container::get_instance().erase("x");
		throw;
	}

	product_table_container::get_instance().erase("x");

}

/** \test Selecting elements from random block tensor (1 block)
 **/
template<typename ComparePolicy>
void btod_select_test::test_1(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_1(size_t)";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 3; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);

	//	Fill in random data
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	// Form list
	ComparePolicy cmp;
	typename btod_select_t::list_t btlist;
	btod_select_t(bt, cmp).perform(btlist, n);

	// Form reference list
	typename tod_select_t::list_t tlist;
	tod_select_t(t_ref, cmp).perform(tlist, n);

	// Check result lists
	typename tod_select_t::list_t::const_iterator it = tlist.begin();
	typename btod_select_t::list_t::const_iterator ibt = btlist.begin();
	while (it != tlist.end() && ibt != btlist.end()) {
		if (it->value != ibt->value) {
			std::ostringstream oss;
			oss << "Value of list element does not match reference "
					<< "(found: " << ibt->value
					<< ", expected: " << it->value << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		index<2> idx = bis.get_block_start(ibt->bidx);
		idx[0] += ibt->idx[0];
		idx[1] += ibt->idx[1];
		if (! idx.equals(it->idx)) {
			std::ostringstream oss;
			oss << "Index of list element does not match reference "
					<< "(found: " << idx
					<< ", expected: " << it->idx << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		it++; ibt++;
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/** \test Selecting elements from random block tensor (multiple blocks)
 **/
template<typename ComparePolicy>
void btod_select_test::test_2(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_2(size_t)";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 5; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10; m01[1] = true; m10[0] = true;
	bis.split(m10, 3);
	bis.split(m01, 4);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);

	//	Fill in random data
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	// Compute list
	ComparePolicy cmp;
	typename btod_select_t::list_t btlist;
	btod_select_t(bt, cmp).perform(btlist, n);

	typename tod_select_t::list_t tlist;
	tod_select_t(t_ref, cmp).perform(tlist, n);

	typename tod_select_t::list_t::const_iterator it = tlist.begin();
	typename btod_select_t::list_t::const_iterator ibt = btlist.begin();
	while (it != tlist.end() && ibt != btlist.end()) {
		if (it->value != ibt->value) {
			std::ostringstream oss;
			oss << "Value of list element does not match reference "
					<< "(found: " << ibt->value
					<< ", expected: " << it->value << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		index<2> idx = bis.get_block_start(ibt->bidx);
		idx[0] += ibt->idx[0];
		idx[1] += ibt->idx[1];
		if (! idx.equals(it->idx)) {
			std::ostringstream oss;
			oss << "Index of list element does not match reference "
					<< "(found: " << idx
					<< ", expected: " << it->idx << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		it++; ibt++;
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}


}

/** \test Selecting elements from random block tensor with symmetry
 **/
template<typename ComparePolicy>
void btod_select_test::test_3(size_t n,
		bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_3(size_t)";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 8; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11; m11[0] = true; m11[1] = true;
	bis.split(m11, 3);
	bis.split(m11, 5);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	{
	block_tensor_ctrl<2, double> ctrl(bt);
	ctrl.req_symmetry().insert(
			se_perm<2, double>(permutation<2>().permute(0, 1), symm));
	}

	//	Fill in random data
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	// Compute list
	ComparePolicy cmp;
	typename btod_select_t::list_t btlist;
	btod_select_t(bt, cmp).perform(btlist, n);

	// Compute reference list
	typename tod_select_t::list_t tlist;
	tod_select_t(t_ref, cmp).perform(tlist, n);

	// Compare against reference
	double last_value = 0.0;
	for (typename btod_select_t::list_t::const_iterator ibt = btlist.begin();
			ibt != btlist.end(); ibt++) {

		typename tod_select_t::list_t::const_iterator it = tlist.begin();
		while (it != tlist.end() && it->value != ibt->value) it++;

		if (it == tlist.end()) {
			std::ostringstream oss;
			oss << "List element not found in reference "
					<< "(" << ibt->bidx << ", "
					<< ibt->idx << ": " << ibt->value << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		while (it != tlist.end() && it->value == ibt->value) {

			index<2> idx = bis.get_block_start(ibt->bidx);
			idx[0] += ibt->idx[0];
			idx[1] += ibt->idx[1];
			if (idx.equals(it->idx)) break;

			it++;
		}

		if (it->value != ibt->value) {
			std::ostringstream oss;
			oss << "List element not found in reference "
					<< "(" << ibt->bidx << ", "
					<< ibt->idx << ": " << ibt->value << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		if (ibt != btlist.begin() && cmp(ibt->value, last_value)) {
			std::ostringstream oss;
			oss << "Invalid ordering of values "
					<< "(" << last_value << " before " << ibt->value
					<< " in list).";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());

		}
		last_value = ibt->value;
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}


}

/** \test Selecting elements from random block tensor without symmetry,
 	 but permutational symmetry imposed.
 **/
template<typename ComparePolicy>
void btod_select_test::test_4(size_t n,
		bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_4(size_t)";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 8; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11; m11[1] = true; m11[0] = true;
	bis.split(m11, 3);
	bis.split(m11, 5);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);

	symmetry<2, double> sym(bis);
	{ // Setup symmetries
	block_tensor_ctrl<2, double> ctrl(bt_ref);
	se_perm<2, double> se(permutation<2>().permute(0, 1), symm);
	sym.insert(se);
	ctrl.req_symmetry().insert(se);
	}

	//	Fill in random data
	btod_random<2>().perform(bt_ref);
	{
	tensor<2, double, allocator_t> tmp(dims);
	tod_btconv<2>(bt_ref).perform(tmp);
	tensor_ctrl<2, double> ctrl(tmp);
	const double *ptr = ctrl.req_const_dataptr();
	btod_import_raw<2>(ptr, dims).perform(bt);
	ctrl.ret_dataptr(ptr);
	}

	// Compute list
	ComparePolicy cmp;
	typename btod_select_t::list_t btlist;
	btod_select_t(bt, sym, cmp).perform(btlist, n);

	// Compute reference list
	typename btod_select_t::list_t btlist_ref;
	btod_select_t(bt_ref, cmp).perform(btlist_ref, n);

	// Compare against reference
	for (typename btod_select_t::list_t::const_iterator ibt = btlist.begin(),
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
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}


}

/** \test Selecting elements from random block tensor without symmetry,
 	 but irrep symmetry imposed.
 **/
template<typename ComparePolicy>
void btod_select_test::test_5(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_5()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 5; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10; m01[1] = true; m10[0] = true;
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

	//	Fill in random data
	btod_random<2>().perform(bt);

	// Compute list
	ComparePolicy cmp;
	typename btod_select_t::list_t btlist;
	btod_select_t(bt, sym, cmp).perform(btlist, n);

	double last_value = 0.0;
	index<2> vidx1, vidx2;
	vidx1[0] = 0; vidx1[1] = 1;
	vidx2[0] = 1; vidx2[1] = 0;

	for (typename btod_select_t::list_t::const_iterator ibt = btlist.begin();
			ibt != btlist.end(); ibt++) {

		if (! ibt->bidx.equals(vidx1) && ! ibt->bidx.equals(vidx2)) {
			std::ostringstream oss;
			oss << "Block index of list element not valid "
					<< "(found: " << ibt->bidx << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		if (ibt != btlist.begin() && cmp(ibt->value, last_value)) {
			std::ostringstream oss;
			oss << "Invalid ordering of values "
					<< "(" << last_value << " before " << ibt->value
					<< " in list).";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}
		last_value = ibt->value;
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/** \test Selecting elements from random block tensor with partition symmetry.
 **/
template<typename ComparePolicy>
void btod_select_test::test_6(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_6()";

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 5; i2[1] = 7;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10; m01[1] = true; m10[0] = true;
	bis.split(m10, 3);
	bis.split(m01, 4);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	{
	block_tensor_ctrl<2, double> cb(bt);
	symmetry<2, double> &sym = cb.req_symmetry();
	mask<2> m; m[0] = true; m[1] = true;
	se_part<2, double> spx(bis, m, 2);
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	spx.add_map(i00, i11);
	spx.add_map(i01, i10);
	sym.insert(spx);
	}

	//	Fill in random data
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	// Compute list
	ComparePolicy cmp;
	typename btod_select_t::list_t btlist;
	btod_select_t(bt, cmp).perform(btlist, n);

	// Compute reference
	typename tod_select_t::list_t tlist;
	tod_select_t(t_ref, cmp).perform(tlist, n);

	// Compare against reference
	double last_value = 0.0;
	for (typename btod_select_t::list_t::const_iterator ibt = btlist.begin();
			ibt != btlist.end(); ibt++) {

		typename tod_select_t::list_t::const_iterator it = tlist.begin();
		while (it != tlist.end() && it->value != ibt->value) it++;

		if (it == tlist.end()) {
			std::ostringstream oss;
			oss << "List element not found in reference "
					<< "(" << ibt->bidx << ", "
					<< ibt->idx << ": " << ibt->value << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		while (it != tlist.end() && it->value == ibt->value) {

			index<2> idx = bis.get_block_start(ibt->bidx);
			idx[0] += ibt->idx[0];
			idx[1] += ibt->idx[1];
			if (idx.equals(it->idx)) break;

			it++;
		}

		if (it->value != ibt->value) {
			std::ostringstream oss;
			oss << "List element not found in reference "
					<< "(" << ibt->bidx << ", "
					<< ibt->idx << ": " << ibt->value << ").";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}

		if (ibt != btlist.begin() && cmp(ibt->value, last_value)) {
			std::ostringstream oss;
			oss << "Invalid ordering of values "
					<< "(" << last_value << " before " << ibt->value
					<< " in list).";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());

		}
		last_value = ibt->value;
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
