#include <sstream>
#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
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
#include <libtensor/tod/tod_copy.h>
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

	test_3a<compare4absmax>(4, true);
	test_3a<compare4absmin>(8, false);
	test_3a<compare4max>(16, false);
	test_3a<compare4min>(32, true);

	test_3b<compare4absmax>(5);
	test_3b<compare4absmin>(10);
	test_3b<compare4max>(13);
	test_3b<compare4min>(11);

	test_3c<compare4absmax>(6, false);
	test_3c<compare4absmin>(7, true);
	test_3c<compare4max>(12, true);
	test_3c<compare4min>(22, false);

	test_4a<compare4absmax>(5, true);
	test_4a<compare4absmin>(10, false);
	test_4a<compare4max>(15, true);
	test_4a<compare4min>(30, false);

	test_4b<compare4absmax>(9);
	test_4b<compare4absmin>(16);
	test_4b<compare4max>(21);
	test_4b<compare4min>(14);

	test_4c<compare4absmax>(7, true);
	test_4c<compare4absmin>(17, false);
	test_4c<compare4max>(25, false);
	test_4c<compare4min>(33, true);

	test_5<compare4absmax>(7);
	test_5<compare4absmin>(12);
	test_5<compare4max>(19);
	test_5<compare4min>(3);

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

	typedef std_allocator<double> allocator_t;
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

	typedef std_allocator<double> allocator_t;
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

/** \test Selecting elements from random block tensor with permutation
		symmetry
 **/
template<typename ComparePolicy>
void btod_select_test::test_3a(size_t n,
		bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_3a(size_t)";

	typedef std_allocator<double> allocator_t;
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
		block_tensor_ctrl<2, double> btc(bt);
		btc.req_symmetry().insert(
			se_perm<2, double>(permutation<2>().permute(0, 1), symm));
	}

	//	Fill in random data
	btod_random<2>().perform(bt);
	{
		block_tensor<2, double, allocator_t> btmp(bis);

		block_tensor_ctrl<2, double> ca(bt), cb(btmp);
		orbit_list<2, double> ol(ca.req_const_symmetry());
		for (orbit_list<2, double>::iterator it = ol.begin();
				it != ol.end(); it++) {

			tensor_i<2, double> &ta = ca.req_block(ol.get_index(it)),
					&tb = cb.req_block(ol.get_index(it));

			tod_copy<2>(ta).perform(tb);

			ca.ret_block(ol.get_index(it));
			cb.ret_block(ol.get_index(it));
		}

		tod_btconv<2>(btmp).perform(t_ref);
	}

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

/** \test Selecting elements from random block tensor with label symmetry
 **/
template<typename ComparePolicy>
void btod_select_test::test_3b(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_3b(size_t)";

	typedef std_allocator<double> allocator_t;
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

	{ // Setup symmetry
	block_tensor_ctrl<2, double> btc(bt);
	se_label<2, double> se(bis.get_block_index_dims(), "x");
	se.assign(m11, 0, 0);
	se.assign(m11, 1, 0);
	se.assign(m11, 2, 1);
	se.add_target(1);
	btc.req_symmetry().insert(se);
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

/** \test Selecting elements from random block tensor with partition symmetry
 **/
template<typename ComparePolicy>
void btod_select_test::test_3c(size_t n,
		bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_3c(size_t, bool)";

	typedef std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11; m11[0] = true; m11[1] = true;
	bis.split(m11, 5);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	{ // Setup symmetry
		block_tensor_ctrl<2, double> btc(bt);
		se_part<2, double> sp(bis, m11, 2);
		sp.add_map(i00, i01, symm);
		sp.add_map(i01, i10, symm);
		sp.add_map(i10, i11, symm);
		btc.req_symmetry().insert(sp);
	}

	//	Fill in random data
	btod_random<2>().perform(bt);
	{
		tensor<2, double, allocator_t> tmp(dims);
		block_tensor<2, double, allocator_t> btmp(bis);

		tod_btconv<2>(bt).perform(tmp);
		tensor_ctrl<2, double> tc(tmp);
		const double *ptr = tc.req_const_dataptr();
		btod_import_raw<2>(ptr, dims).perform(btmp);
		tc.ret_const_dataptr(ptr);

		block_tensor_ctrl<2, double> btc(btmp);
		btc.req_zero_block(i01);
		btc.req_zero_block(i10);
		btc.req_zero_block(i11);
		tod_btconv<2>(btmp).perform(t_ref);
	}

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
void btod_select_test::test_4a(size_t n,
		bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_4a(size_t, bool)";

	typedef std_allocator<double> allocator_t;
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
		se_perm<2, double> se(permutation<2>().permute(0, 1), symm);
		sym.insert(se);

		block_tensor_ctrl<2, double> ctrl(bt_ref);
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
	ctrl.ret_const_dataptr(ptr);
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
 	 but label symmetry imposed.
 **/
template<typename ComparePolicy>
void btod_select_test::test_4b(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_4(size_t)";

	typedef std_allocator<double> allocator_t;
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
		se_label<2, double> se(bis.get_block_index_dims(), "x");
		se.assign(m11, 0, 0);
		se.assign(m11, 1, 1);
		se.assign(m11, 2, 1);
		se.add_target(0);
		sym.insert(se);

		block_tensor_ctrl<2, double> ctrl(bt_ref);
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
	ctrl.ret_const_dataptr(ptr);
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
		but partition symmetry imposed.
 **/
template<typename ComparePolicy>
void btod_select_test::test_4c(size_t n,
		bool symm) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_4c(size_t, bool)";

	typedef std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 7; i2[1] = 7;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11; m11[1] = true; m11[0] = true;
	bis.split(m11, 4);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	symmetry<2, double> sym(bis);

	{ // Setup symmetries
		se_part<2, double> sp(bis, m11, 2);
		index<2> i00, i01, i10, i11;
		i10[0] = 1; i01[1] = 1;
		i11[0] = 1; i11[1] = 1;
		sp.add_map(i00, i11, symm);
		sp.add_map(i01, i10, true);
		sym.insert(sp);

		block_tensor_ctrl<2, double> ctrl(bt_ref);
		ctrl.req_symmetry().insert(sp);
	}

	//	Fill in random data
	btod_random<2>().perform(bt_ref);
	{
	tensor<2, double, allocator_t> tmp(dims);
	tod_btconv<2>(bt_ref).perform(tmp);
	tensor_ctrl<2, double> ctrl(tmp);
	const double *ptr = ctrl.req_const_dataptr();
	btod_import_raw<2>(ptr, dims).perform(bt);
	ctrl.ret_const_dataptr(ptr);
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


/** \test Selecting elements from random block tensor with mixed symmetry
 	 and different symmetry imposed.
 **/
template<typename ComparePolicy>
void btod_select_test::test_5(size_t n) throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_5(size_t)";

	typedef std_allocator<double> allocator_t;
	typedef tod_select<2, ComparePolicy> tod_select_t;
	typedef btod_select<2, ComparePolicy> btod_select_t;

	try {

	index<2> i1, i2; i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m11; m11[1] = true; m11[0] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);
	block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
	symmetry<2, double> sym(bis);

	{
		se_perm<2, double> se1(permutation<2>().permute(0, 1), true),
				se2(permutation<2>().permute(0, 1), false);
		se_label<2, double> sl(bis.get_block_index_dims(), "x");
		se_part<2, double> sp1(bis, m11, 2), sp2(bis, m11, 2);

		sl.assign(m11, 0, 0);
		sl.assign(m11, 1, 1);
		sl.assign(m11, 2, 0);
		sl.assign(m11, 3, 1);
		sl.add_target(1);

		index<2> i00, i01, i10, i11;
		i10[0] = 1; i01[1] = 1;
		i11[0] = 1; i11[1] = 1;
		sp1.add_map(i00, i01, false);
		sp1.add_map(i01, i10, true);
		sp1.add_map(i10, i11, false);
		sp2.add_map(i00, i11, true);

		block_tensor_ctrl<2, double> btc(bt);
		btc.req_symmetry().insert(se1);
		btc.req_symmetry().insert(sl);
		btc.req_symmetry().insert(sp1);

		sym.insert(se2);
		sym.insert(sp2);
	}

	//	Fill in random data
	btod_random<2>().perform(bt);
	{
		block_tensor_ctrl<2, double> ca(bt), cb(bt_ref);
		dimensions<2> bidims(bis.get_block_index_dims());
		cb.req_zero_all_blocks();
		orbit_list<2, double> ol(sym);
		for (orbit_list<2, double>::iterator it = ol.begin();
				it != ol.end(); it++) {

			index<2> ib = ol.get_index(it);
			orbit<2, double> oa(ca.req_const_symmetry(), ib);
			if (! oa.is_allowed()) continue;

			const transf<2, double> &tra = oa.get_transf(ib);

			abs_index<2> ai(oa.get_abs_canonical_index(), bidims);
			tensor_i<2, double> &ta = ca.req_block(ai.get_index()),
					&tb = cb.req_block(ib);
			tod_copy<2>(ta, tra.get_perm(), tra.get_coeff()).perform(tb);

			ca.ret_block(ai.get_index());
			cb.ret_block(ib);
		}
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


} // namespace libtensor
