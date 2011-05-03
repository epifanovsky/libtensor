#include <libtensor/symmetry/so_apply_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "so_apply_impl_perm_test.h"

namespace libtensor {


void so_apply_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1(false, false);
	test_1(false, true);
	test_1(true, false);
	test_2(false, false);
	test_2(false, true);
	test_2(true, false);
	test_3(false, false);
	test_3(false, true);
	test_3(true, false);
}


/**	\test Tests that an empty sets yields an empty set
 **/
void so_apply_impl_perm_test::test_1(
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_apply_impl_perm_test::test_1("
			<< (is_asym ? "true" : "false") << ", "
			<< (sign ? "true" : "false") << ")";

	typedef se_perm<2, double> se_t;
	typedef so_apply<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_apply_impl_perm_t;

	try {

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);

	permutation<2> p0;
	params_t params(set1, p0, is_asym, sign, set2);

	so_apply_impl_perm_t op;
	op.perform(params);
	if(!set2.is_empty()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
				"!set2.is_empty() (1).");
	}

	permutation<2> perm; perm.permute(0, 1);
	se_t elem(perm, true);
	set2.insert(elem);

	op.perform(params);
	if(!set2.is_empty()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
			"!set2.is_empty() (2).");
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the application on a non-empty set
 **/
void so_apply_impl_perm_test::test_2(
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_apply_impl_perm_test::test_2("
			<< (is_asym ? "true" : "false") << ", "
			<< (sign ? "true" : "false") << ")";

	typedef se_perm<2, double> se_t;
	typedef so_apply<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_apply_impl_perm_t;

	try {

	se_t elem1(permutation<2>().permute(0, 1), false);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);

	set1.insert(elem1);

	permutation<2> p0;
	params_t params(set1, p0, is_asym, sign, set2);

	so_apply_impl_perm_t op;
	op.perform(params);

	if (is_asym) {
		if(! set2.is_empty())
			fail_test(tnss.str().c_str(),
					__FILE__, __LINE__, "! set2.is_empty()");
	}
	else {
		if(set2.is_empty())
			fail_test(tnss.str().c_str(),
					__FILE__, __LINE__, "set2.is_empty()");

		symmetry_element_set_adapter<2, double, se_t> adapter(set2);
		symmetry_element_set_adapter<2, double, se_t>::iterator i =
				adapter.begin();
		const se_t &elem2 = adapter.get_elem(i);
		i++;
		if(i != adapter.end())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Expected only one element.");

		if (sign) {
			if (! elem2.is_symm())
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"!elem2.is_symm()");
		}
		else {
			if(elem2.is_symm())
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"elem2.is_symm()");
		}

		if(!elem1.get_perm().equals(elem2.get_perm()))
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"elem1 != elem2");
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests the application on a non-empty set with permutation
 **/
void so_apply_impl_perm_test::test_3(
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_apply_impl_perm_test::test_3("
			<< (is_asym ? "true" : "false") << ", "
			<< (sign ? "true" : "false") << ")";

	typedef se_perm<4, double> se_t;
	typedef so_apply<4, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;
	typedef symmetry_operation_impl<so_t, se_t> so_apply_impl_perm_t;

	try {

	se_t el1(permutation<4>().permute(0, 1), false);
	se_t el2(permutation<4>().permute(2, 3), true);

	symmetry_element_set<4, double> set1(se_t::k_sym_type);
	symmetry_element_set<4, double> set2(se_t::k_sym_type);

	set1.insert(el1);
	set1.insert(el2);

	permutation<4> perm; perm.permute(0, 1).permute(1, 2);
	params_t params(set1, perm, is_asym, sign, set2);

	so_apply_impl_perm_t op;
	op.perform(params);

	if (is_asym) {
		if(! set2.is_empty())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"! set2.is_empty()");
	}
	else {
		if(set2.is_empty())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"set2.is_empty()");

		symmetry_element_set_adapter<4, double, se_t> adapter(set2);
		symmetry_element_set_adapter<4, double, se_t>::iterator i =
				adapter.begin();
		const se_t &elem1 = adapter.get_elem(i); i++;
		if(i == adapter.end())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Expected two elements.");

		const se_t &elem2 = adapter.get_elem(i); i++;
		if(i != adapter.end())
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Expected only two elements.");

		bool is_p02;
		if (elem1.get_perm().equals(permutation<4>().permute(0, 2))) {
			is_p02 = true;
			if (sign) {
				if (! elem1.is_symm())
					fail_test(tnss.str().c_str(), __FILE__, __LINE__,
							"!elem1.is_symm()");
			}
			else {
				if (elem1.is_symm())
					fail_test(tnss.str().c_str(), __FILE__, __LINE__,
							"elem1.is_symm()");
			}
		}
		else if (elem1.get_perm().equals(permutation<4>().permute(1, 3))) {
			is_p02 = false;
			if (! elem1.is_symm())
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"!elem1.is_symm()");
		}
		else {
			fail_test(tnss.str().c_str(), __FILE__, __LINE__,
					"Unexpected permutation in elem1.");
		}

		if (is_p02) {
			if (! elem2.get_perm().equals(permutation<4>().permute(1, 3)))
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"Unexpected permutation elem2.");

			if (! elem2.is_symm())
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"!elem2.is_symm()");
		}
		else {
			if (! elem2.get_perm().equals(permutation<4>().permute(0, 2)))
				fail_test(tnss.str().c_str(), __FILE__, __LINE__,
						"Unexpected permutation elem2.");

			if (sign) {
				if (! elem2.is_symm())
					fail_test(tnss.str().c_str(), __FILE__, __LINE__,
							"!elem2.is_symm()");
			}
			else {
				if(elem2.is_symm())
					fail_test(tnss.str().c_str(), __FILE__, __LINE__,
							"elem2.is_symm()");
			}
		}
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
