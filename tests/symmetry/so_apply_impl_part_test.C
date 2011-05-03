#include <libtensor/symmetry/so_apply_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_apply_impl_part_test.h"

namespace libtensor {


void so_apply_impl_part_test::perform() throw(libtest::test_exception) {

	test_1( true, false);
	test_1(false, false);
	test_1(false,  true);
	test_2( true, false);
	test_2(false, false);
	test_2(false,  true);
	test_3( true, false);
	test_3(false, false);
	test_3(false,  true);
}


/**	\test Tests that application on an empty set yields an empty set
 **/
void so_apply_impl_part_test::test_1(
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss <<  "so_apply_impl_part_test::test_1("
			<< (is_asym ? "true" : "false") << ", "
			<< (sign ? "true" : "false") << ")";

	typedef se_part<2, double> se_t;
	typedef so_apply<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_impl_part_t;

	try {

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);

	permutation<2> p0;
	params_t params(set1, p0, is_asym, sign, set2);

	so_impl_part_t op;

	op.perform(params);
	if(!set2.is_empty()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
			"!set2.is_empty() (1).");
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
	set2.insert(elem);

	op.perform(params);
	if(!set2.is_empty())
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
				"!set2.is_empty() (2).");

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests application on a non-empty set.
 **/
void so_apply_impl_part_test::test_2(
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss <<  "so_apply_impl_part_test::test_2("
			<< (is_asym ? "true" : "false") << ", "
			<< (sign ? "true" : "false") << ")";

	typedef se_part<2, double> se_t;
	typedef so_apply<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_impl_part_t;

	try {

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 5;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 3);

	index<2> i00, i11, i01, i10;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	se_t elem1(bis, m11, 2);
	elem1.add_map(i00, i11, true);
	elem1.add_map(i01, i10, false);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);

	set1.insert(elem1);

	permutation<2> p0;
	params_t params(set1, p0, is_asym, sign, set2);

	so_impl_part_t op;
	op.perform(params);

	if (is_asym) {
		if(! set2.is_empty())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"! set2.is_empty().");
	}
	else {
		if(set2.is_empty())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"set2.is_empty().");

		symmetry_element_set_adapter<2, double, se_t> ad2(set2);
		symmetry_element_set_adapter<2, double, se_t>::iterator it = ad2.begin();

		const se_t &elem2 = ad2.get_elem(it);
		it++;
		if(it != ad2.end())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Expected only one element.");

		if (! elem2.map_exists(i00, i11))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Map [0, 0]->[1, 1] does not exist.");

		if (! elem2.get_sign(i00, i11))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Wrong sign of map [0, 0]->[1, 1].");

		if (! elem2.map_exists(i01, i10))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Map [0, 1]->[1, 0] does not exist.");

		if (sign) {
			if (! elem2.get_sign(i01, i10))
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"Wrong sign of map [0, 0]->[1, 1].");
		}
		else {
			if (elem2.get_sign(i01, i10))
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"Wrong sign of map [0, 0]->[1, 1].");
		}
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests application on non-empty set with permutation
 **/
void so_apply_impl_part_test::test_3(
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss <<  "so_apply_impl_part_test::test_2("
			<< (is_asym ? "true" : "false") << ", "
			<< (sign ? "true" : "false") << ")";

	typedef se_part<2, double> se_t;
	typedef so_apply<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_impl_part_t;

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
	elem1.add_map(i00, i01, true);
	elem1.add_map(i10, i11, false);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);

	set1.insert(elem1);

	permutation<2> p1; p1.permute(0, 1);
	params_t params(set1, p1, is_asym, sign, set2);

	so_impl_part_t op;
	op.perform(params);

	if (is_asym) {
		if (! set2.is_empty())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"!set2.is_empty().");
	}
	else {
		if (set2.is_empty())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"set2.is_empty().");

		symmetry_element_set_adapter<2, double, se_t> ad2(set2);
		symmetry_element_set_adapter<2, double, se_t>::iterator it = ad2.begin();

		const se_t &elem2 = ad2.get_elem(it);
		it++;
		if(it != ad2.end())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Expected only one element.");

		if (! elem2.map_exists(i00, i10))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Map [0, 0]->[1, 0] does not exist.");

		if (! elem2.get_sign(i00, i10))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Wrong sign of map [0, 0]->[1, 0].");

		if (! elem2.map_exists(i01, i11))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Map [0, 1]->[1, 1] does not exist.");

		if (sign) {
			if (! elem2.get_sign(i01, i11))
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"Wrong sign of map [0, 0]->[1, 0].");
		}
		else {
			if (elem2.get_sign(i01, i11))
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"Wrong sign of map [0, 0]->[1, 0].");
		}
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
