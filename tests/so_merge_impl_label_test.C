#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/so_merge_impl_label.h>
#include <libtensor/btod/transf_double.h>
#include "so_merge_impl_label_test.h"

namespace libtensor {

const char *so_merge_impl_label_test::k_table_id = "point_group";

void so_merge_impl_label_test::perform() throw(libtest::test_exception) {

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
	test_4();

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(k_table_id);
		throw;
	}

	product_table_container::get_instance().erase(k_table_id);
}


/**	\test Tests that merge of 2 dim of an empty group yields an empty group
		of a lower order
 **/
void so_merge_impl_label_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_label_test::test_1()";

	typedef se_label<4, double> se4_t;
	typedef se_label<3, double> se2_t;
	typedef so_merge<4, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<3, double> set2(se2_t::k_sym_type);

	mask<4> msk; msk[2] = true; msk[3] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 2 dim of a 2-space on a 1-space.
 **/
void so_merge_impl_label_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_label_test::test_2()";

	typedef se_label<1, double> se1_t;
	typedef se_label<2, double> se2_t;
	typedef so_merge<2, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se2_t>
		so_merge_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 3; i2b[1] = 3;
	dimensions<2> dim2(index_range<2>(i2a, i2b));
	se2_t elem2(dim2, k_table_id);
	mask<2> m;
	m[0] = true; m[1] = true;

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m, i, i);
	elem2.add_target(2);

	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<1, double> set1(se1_t::k_sym_type);
	symmetry_element_set<1, double> set1_ref(se1_t::k_sym_type);

	set2.insert(elem2);
	symmetry_operation_params<so_merge_t> params(set2, m, set1);

	so_merge_impl_t().perform(params);

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
	if(elem.get_n_targets() != 4) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 4");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 3 dim of a 5-space onto a 3-space.
 **/
void so_merge_impl_label_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_label_test::test_3()";

	typedef se_label<3, double> se3_t;
	typedef se_label<5, double> se5_t;
	typedef so_merge<5, 3, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se5_t>
		so_merge_impl_t;

	try {

	index<5> i5a, i5b;
	i5b[0] = 3; i5b[1] = 3; i5b[2] = 3; i5b[3] = 3; i5b[4] = 3;
 	dimensions<5> dim5(index_range<5>(i5a, i5b));
	se5_t elem5(dim5, k_table_id);
	mask<5> m5a, m5b;
	m5a[0] = true; m5a[4] = true; m5b[1] = true; m5b[2] = true; m5b[3] = true;
	point_group_table::label_t map[4];
	map[0] = 1; map[1] = 3; map[2] = 0; map[3] = 2;

	for (size_t i = 0; i < 4; i++) {
		elem5.assign(m5a, i, i);
		elem5.assign(m5b, i, map[i]);
	}
	elem5.add_target(2);

	symmetry_element_set<5, double> set1(se5_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	set1.insert(elem5);

	symmetry_operation_params<so_merge_t> params(set1, m5b, set2);

	so_merge_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem.get_n_targets() != 4) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 4");
	}
	if(elem.get_label(elem.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Merge of 2 dim of a 4-space onto a 3-space, different dimensions.
 **/
void so_merge_impl_label_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_label_test::test_4()";

	typedef se_label<3, double> se3_t;
	typedef se_label<4, double> se4_t;
	typedef so_merge<4, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 3; i4b[1] = 5; i4b[2] = 3; i4b[3] = 5;
 	dimensions<4> dim4(index_range<4>(i4a, i4b));
	se4_t elema(dim4, k_table_id);
	mask<4> m4a, m4b;
	m4a[0] = true; m4b[1] = true; m4a[2] = true; m4b[3] = true;
	point_group_table::label_t map[6];
	map[0] = 2; map[1] = 0; map[2] = 1; map[3] = 2; map[4] = 3; map[5] = 3;

	for (size_t i = 0; i < 4; i++) elema.assign(m4a, i, i);
	for (size_t i = 0; i < 6; i++) elema.assign(m4b, i, map[i]);
	elema.add_target(2);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	set1.insert(elema);

	symmetry_operation_params<so_merge_t> params(set1, m4b, set2);

	so_merge_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem.get_n_targets() != 4) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 4");
	}
	if(elem.get_label(elem.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem.get_label(elem.get_dim_type(1), 0) != 2) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(1, 0) != Au");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
