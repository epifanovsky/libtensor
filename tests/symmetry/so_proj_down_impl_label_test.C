#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/so_proj_down_impl_label.h>
#include <libtensor/btod/transf_double.h>
#include "so_proj_down_impl_label_test.h"

namespace libtensor {

const char *so_proj_down_impl_label_test::k_table_id = "point_group";

void so_proj_down_impl_label_test::perform() throw(libtest::test_exception) {

	try {

	point_group_table s6(k_table_id, 4);
	point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
	s6.add_product(ag, ag, ag);
	s6.add_product(ag, eg, eg);
	s6.add_product(ag, au, au);
	s6.add_product(ag, eu, eu);
	s6.add_product(eg, eg, ag);
	s6.add_product(eg, eg, eg);
	s6.add_product(eg, au, eu);
	s6.add_product(eg, eu, au);
	s6.add_product(eg, eu, eu);
	s6.add_product(au, au, ag);
	s6.add_product(au, eu, eg);
	s6.add_product(eu, eu, ag);
	s6.add_product(eu, eu, eg);
	s6.check();
	product_table_container::get_instance().add(s6);

	} catch (exception &e) {
		fail_test("so_mult_impl_perm_test::perform()", __FILE__, __LINE__,
				e.what());
	}

	try {

	test_1();
	test_2();
	test_3();

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(k_table_id);
		throw;
	}

	product_table_container::get_instance().erase(k_table_id);
}


/**	\test Tests that a projection of an empty group yields an empty group
		of a lower order
 **/
void so_proj_down_impl_label_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_label_test::test_1()";

	typedef se_label<3, double> se3_t;
	typedef se_label<2, double> se2_t;
	typedef so_proj_down<3, 1, double> so_proj_down_t;
	typedef symmetry_operation_impl<so_proj_down_t, se3_t>
		so_proj_down_impl_t;

	try {

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	mask<3> msk; msk[0] = true; msk[1] = true; msk[2] = false;
	symmetry_operation_params<so_proj_down_t> params(set1, msk, set2);

	so_proj_down_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of a 2-space on a 1-space.
 **/
void so_proj_down_impl_label_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_label_test::test_2()";

	typedef se_label<1, double> se1_t;
	typedef se_label<2, double> se2_t;
	typedef so_proj_down<2, 1, double> so_proj_down_t;
	typedef symmetry_operation_impl<so_proj_down_t, se2_t>
		so_proj_down_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 3; i2b[1] = 3;
	dimensions<2> dim2(index_range<2>(i2a, i2b));
	se2_t elem2(dim2, k_table_id);
	mask<2> m2a, m2b;
	m2a[0] = true; m2a[1] = false; m2b[0] = false; m2b[1] = true;
	point_group_table::label_t map[4];
	map[0] = 2; map[1] = 0; map[2] = 3; map[3] = 1;

	for (unsigned int i = 0; i < 4; i++) {
		elem2.assign(m2a, i, i);
		elem2.assign(m2b, i, map[i]);
	}
	elem2.add_target(2);

	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<1, double> set1(se1_t::k_sym_type);
	symmetry_element_set<1, double> set1_ref(se1_t::k_sym_type);

	set2.insert(elem2);
	symmetry_operation_params<so_proj_down_t> params(set2, m2a, set1);

	so_proj_down_impl_t().perform(params);

	if(set1.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<1, double, se1_t> adapter(set1);
	symmetry_element_set_adapter<1, double, se1_t>::iterator i =
		adapter.begin();
	const se1_t &elem = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem.get_n_targets() != elem.get_n_labels()) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != max");
	}
	if(elem.get_label(elem.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of a 4-space onto a 2-space.
 **/
void so_proj_down_impl_label_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_label_test::test_3()";

	typedef se_label<2, double> se2_t;
	typedef se_label<4, double> se4_t;
	typedef so_proj_down<4, 2, double> so_proj_down_t;
	typedef symmetry_operation_impl<so_proj_down_t, se4_t>
		so_proj_down_impl_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 3; i4b[1] = 3; i4b[2] = 3; i4b[3] = 3;
	dimensions<4> dim4(index_range<4>(i4a, i4b));
	se4_t elem4(dim4, k_table_id);
	mask<4> m4a, m4b, m4c, m4;
	m4a[0] = true; m4b[1] = true; m4c[2] = true; m4c[3] = true;
	m4[0] = true; m4[2] = true;
	point_group_table::label_t mapb[4], mapc[4];
	mapb[0] = 1; mapb[1] = 3; mapb[2] = 0; mapb[3] = 2;
	mapc[0] = 2; mapc[1] = 0; mapc[2] = 3; mapc[3] = 1;

	for (size_t i = 0; i < 4; i++) {
		elem4.assign(m4a, i, i);
		elem4.assign(m4b, i, mapb[i]);
		elem4.assign(m4c, i, mapc[i]);
	}
	elem4.add_target(2);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	set1.insert(elem4);
	symmetry_operation_params<so_proj_down_t> params(set1, m4, set2);

	so_proj_down_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
	symmetry_element_set_adapter<2, double, se2_t>::iterator i =
		adapter.begin();
	const se2_t &elem = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem.get_n_targets() != elem.get_n_labels()) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 1");
	}
	if(elem.get_label(elem.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem.get_label(elem.get_dim_type(1), 0) != mapc[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(1, 0) != Au");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}



} // namespace libtensor
