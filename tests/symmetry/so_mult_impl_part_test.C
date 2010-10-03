#include <libtensor/symmetry/so_mult_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_mult_impl_part_test.h"

namespace libtensor {


void so_mult_impl_part_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4(true);
	test_4(false);
}


/**	\test Tests that the multiplication of two empty sets yields an empty set
 **/
void so_mult_impl_part_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_part_test::test_1()";

	typedef se_part<2, double> se_t;
	typedef so_mult<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_mult_impl_part_t;

	try {

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	permutation<2> p0;
	params_t params(set1, p0, set2, p0, set3);

	so_mult_impl_part_t op;

	op.perform(params);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 5;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 3);

	index<2> i00, i11;
	i11[0] = 1; i11[1] = 1;
	se_t elem(bis, m11, 2);
	elem.add_map(i00, i11, true);
	set3.insert(elem);

	op.perform(params);
	if(!set3.is_empty())
		fail_test(testname, __FILE__, __LINE__, "!set3.is_empty() (2).");


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests that the multiplication of an empty set and a non-empty set
		yields an empty set
 **/
void so_mult_impl_part_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_part_test::test_2()";

	typedef se_part<2, double> se_t;
	typedef so_mult<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_mult_impl_part_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 5;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 3);

	index<2> i00, i11;
	i11[0] = 1; i11[1] = 1;
	se_t elem1(bis, m11, 2);
	elem1.add_map(i00, i11, true);


	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	set1.insert(elem1);

	permutation<2> p0;
	params_t params(set1, p0, set2, p0, set3);

	so_mult_impl_part_t op;
	op.perform(params);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "!set3.is_empty() (1).");
	}

	set3.insert(elem1);
	op.perform(params);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests that the multiplication of two non-overlapping non-empty sets
		yields an empty set
 **/
void so_mult_impl_part_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_part_test::test_3()";

	typedef se_part<2, double> se_t;
	typedef so_mult<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_mult_impl_part_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 5;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11, m01, m10;
	m10[0] = true; m01[1] = true;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 3);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_t elem1(bis, m01, 2), elem2(bis, m10, 2);
	elem1.add_map(i00, i01, true);
	elem2.add_map(i00, i10, true);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem2);

	permutation<2> p0;
	params_t params(set1, p0, set2, p0, set3);

	so_mult_impl_part_t op;
	op.perform(params);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	set3.insert(elem1);
	set3.insert(elem2);
	op.perform(params);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (2).");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the multiplication of two identical non-empty sets
 **/
void so_mult_impl_part_test::test_4(bool sign) throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_part_test::test_4(bool)";

	typedef se_part<2, double> se_t;
	typedef so_mult<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_mult_impl_part_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 5;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 3);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	se_t elem1(bis, m11, 2);
	elem1.add_map(i00, i11, sign);
	elem1.add_map(i01, i10, sign);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem1);

	set3_ref.insert(elem1);

	permutation<2> p0;
	params_t params(set1, p0, set2, p0, set3);

	so_mult_impl_part_t op;
	op.perform(params);
	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "set3.is_empty()");
	}

	compare_ref<2>::compare(testname, bis, set3, set3_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
