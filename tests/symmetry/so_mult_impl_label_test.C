#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_mult_impl_label.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_mult_impl_label_test.h"

namespace libtensor {

const char *so_mult_impl_label_test::k_table_id = "point_group";

void so_mult_impl_label_test::perform() throw(libtest::test_exception) {

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
	test_2a();
	test_2b();
	test_3a();
	test_3b();
	test_4();
	test_5a();
	test_5b();
	test_5c();

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(k_table_id);
		throw;
	}

	product_table_container::get_instance().erase(k_table_id);

}


/**	\test Tests that the multiplication of two empty groups yields also an
		empty group

 **/
void so_mult_impl_label_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_1()";

	typedef se_label<2, double> se2_t;
	typedef so_mult<2, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se2_t> so_mult_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i2a, i2b)));
	mask<2> m2;
	m2[0] = true; m2[1] = true;
	bis.split(m2, 2); bis.split(m2, 4); bis.split(m2, 6);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se2_t::k_sym_type);

	symmetry_operation_params< so_mult_t > params(
			set1, permutation<2>(), set2, permutation<2>(), set3);

	so_mult_impl_t().perform(params);

	compare_ref<2>::compare(testname, bis, set3, set3_ref);

	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Multiplication of a group with one element of Au symmetry and an
		empty group (2-space). The result is expected to contain one element
		of Au symmetry.
 **/
void so_mult_impl_label_test::test_2a() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_2a(bool)";

	typedef se_label<2, double> se2_t;
	typedef so_mult<2, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se2_t> so_mult_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);

	se2_t elem2(bis2.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m2, i, i);
	elem2.add_target(2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se2_t::k_sym_type);

	set1.insert(elem2);
	set3_ref.insert(elem2);

	symmetry_operation_params<so_mult_t> params(
		set1, permutation<2>(), set2, permutation<2>(), set3);

	so_mult_impl_t().perform(params);

	compare_ref<2>::compare(testname, bis2, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Multiplication of an empty group and a group with one element of
		Au symmetry (2-space). The result is expected to contain no elements.
 **/
void so_mult_impl_label_test::test_2b() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_2b()";

	typedef se_label<2, double> se2_t;
	typedef so_mult<2, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se2_t> so_mult_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);

	se2_t elem2(bis2.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m2, i, i);
	elem2.add_target(2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se2_t::k_sym_type);

	set2.insert(elem2);
	set3_ref.insert(elem2);

	symmetry_operation_params<so_mult_t> params(
		set1, permutation<2>(), set2, permutation<2>(), set3);

	so_mult_impl_t().perform(params);

	compare_ref<2>::compare(testname, bis2, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Multiplication of two groups with one element of Au symmetry each
		(2-space). The result is expected to contain an element of Au symmetry.
 **/
void so_mult_impl_label_test::test_3a() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_3a()";

	typedef se_label<2, double> se2_t;
	typedef so_mult<2, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se2_t> so_mult_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);

	se2_t elem2(bis2.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m2, i, i);

	elem2.add_target(2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se2_t::k_sym_type);

	set1.insert(elem2);
	set2.insert(elem2);
	set3_ref.insert(elem2);

	symmetry_operation_params<so_mult_t> params(
		set1, permutation<2>(), set2, permutation<2>(), set3);

	so_mult_impl_t().perform(params);

	compare_ref<2>::compare(testname, bis2, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<2, double, se2_t> adapter(set3);
	symmetry_element_set_adapter<2, double, se2_t>::iterator i =
		adapter.begin();
	const se2_t &elem = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 1");
	}
	if(elem.get_target(0) != 2) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_target(0) != Au");
	}
	if(elem.get_dim_type(0) != elem.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) != elem.get_dim_type(1)");
	}
	if(elem.get_label(elem.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Multiplication of a group with one element of Au symmetry and a group
		with one element of Eg symmetry (2-space). The result is expected to
		contain an element without target symmetry.
 **/
void so_mult_impl_label_test::test_3b() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_3b()";

	typedef se_label<2, double> se2_t;
	typedef so_mult<2, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se2_t> so_mult_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);

	se2_t elem2_1(bis2.get_block_index_dims(), k_table_id);
	for (unsigned int i = 0; i < 4; i++) elem2_1.assign(m2, i, i);
	se2_t elem2_2(elem2_1), elem2_ref(elem2_1);
	elem2_1.add_target(2);
	elem2_2.add_target(1);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se2_t::k_sym_type);

	set1.insert(elem2_1);
	set2.insert(elem2_2);
	set3_ref.insert(elem2_ref);

	symmetry_operation_params<so_mult_t> params(
		set1, permutation<2>(), set2, permutation<2>(), set3);

	so_mult_impl_t().perform(params);

	compare_ref<2>::compare(testname, bis2, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<2, double, se2_t> adapter(set3);
	symmetry_element_set_adapter<2, double, se2_t>::iterator i =
		adapter.begin();
	const se2_t &elem2_3 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem2_3.get_n_targets() != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 0");
	}
	if(elem2_3.get_dim_type(0) != elem2_3.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) != elem.get_dim_type(1)");
	}
	if(elem2_3.get_label(elem2_3.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Multiplication of two groups with one element each (2-space). The
		labeling does not match.
 **/
void so_mult_impl_label_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_4()";

	typedef se_label<2, double> se2_t;
	typedef so_mult<2, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se2_t> so_mult_impl_t;

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));

 	mask<2> m2, m2a, m2b;
 	m2[0] = true; m2[1] = true;
 	m2a[0] = true; m2b[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);

	se2_t elem2_1(bis2.get_block_index_dims(), k_table_id);
	se2_t elem2_2(bis2.get_block_index_dims(), k_table_id);

	size_t map[4];
	map[0] = 1; map[1] = 3; map[2] = 2; map[3] = 0;
	for (unsigned int i = 0; i < 4; i++) {
		elem2_1.assign(m2a, i, i);
		elem2_1.assign(m2b, i, map[i]);
		elem2_2.assign(m2a, i, map[i]);
		elem2_2.assign(m2b, i, i);
	}

	se2_t elem2_ref(elem2_1);
	elem2_1.add_target(0);
	elem2_2.add_target(0);
	elem2_ref.add_target(0);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3(se2_t::k_sym_type);
	symmetry_element_set<2, double> set3_ref(se2_t::k_sym_type);

	set1.insert(elem2_1);
	set2.insert(elem2_2);
	set3_ref.insert(elem2_ref);

	bool failed = false;
	try {

	symmetry_operation_params<so_mult_t> params(
		set1, permutation<2>(), set2, permutation<2>(), set3);

	so_mult_impl_t().perform(params);

	} catch (exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Illegal labeling stays undetected.");
	}

	try {

	symmetry_operation_params<so_mult_t> params(
		set1, permutation<2>(), set2, permutation<2>().permute(0, 1), set3);

	so_mult_impl_t().perform(params);

	compare_ref<2>::compare(testname, bis2, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<2, double, se2_t> adapter(set3);
	symmetry_element_set_adapter<2, double, se2_t>::iterator i =
		adapter.begin();
	const se2_t &elem2_3 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem2_3.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 1");
	}
	if(elem2_3.get_target(0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_target(0) != Ag");
	}
	if(elem2_3.get_dim_type(0) == elem2_3.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(1)");
	}
	if(elem2_3.get_label(elem2_3.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem2_3.get_label(elem2_3.get_dim_type(1), 0) != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Eg");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Multiplication of a group with one element of Ag + Eg symmetry and
		a group with one element of Eg symmetry (3-space), first group permuted
		via [012->201]. The result is expected to contain an element of
		Eg symmetry.
 **/
void so_mult_impl_label_test::test_5a() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_5a()";

	typedef se_label<3, double> se3_t;
	typedef so_mult<3, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se3_t> so_mult_impl_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 8; i3b[1] = 8; i3b[2] = 8;

	block_index_space<3> bis3(dimensions<3>(index_range<3>(i3a, i3b)));

 	mask<3> m3, m3a, m3b, m3c;
 	m3[0] = true; m3[1] = true; m3[2] = true;
 	m3a[0] = true; m3b[1] = true; m3c[2] = true;
	bis3.split(m3, 2); bis3.split(m3, 4); bis3.split(m3, 6);

	se3_t elem3_1(bis3.get_block_index_dims(), k_table_id);
	se3_t elem3_2(bis3.get_block_index_dims(), k_table_id);

	size_t mapb[4], mapc[4];
	mapb[0] = 1; mapb[1] = 3; mapb[2] = 2; mapb[3] = 0;
	mapc[0] = 2; mapc[1] = 0; mapc[2] = 3; mapc[3] = 1;
	for (unsigned int i = 0; i < 4; i++) {
		elem3_1.assign(m3a, i, i);
		elem3_1.assign(m3b, i, mapb[i]);
		elem3_1.assign(m3c, i, mapc[i]);
		elem3_2.assign(m3a, i, mapc[i]);
		elem3_2.assign(m3b, i, i);
		elem3_2.assign(m3c, i, mapb[i]);
	}

	se3_t elem3_ref(elem3_2);
	elem3_1.add_target(0);
	elem3_1.add_target(1);
	elem3_2.add_target(1);
	elem3_ref.add_target(1);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set3(se3_t::k_sym_type);
	symmetry_element_set<3, double> set3_ref(se3_t::k_sym_type);

	set1.insert(elem3_1);
	set2.insert(elem3_2);
	set3_ref.insert(elem3_ref);

	symmetry_operation_params<so_mult_t> params(
			set1, permutation<3>().permute(1, 2).permute(0, 1),
			set2, permutation<3>(), set3);

	so_mult_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis3, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<3, double, se3_t> adapter(set3);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem3_3 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem3_3.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 1");
	}
	if(elem3_3.get_target(0) != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_target(0) != Eg");
	}
	if(elem3_3.get_dim_type(0) == elem3_3.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(1)");
	}
	if(elem3_3.get_dim_type(0) == elem3_3.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(2)");
	}
	if(elem3_3.get_dim_type(1) == elem3_3.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(1) == elem.get_dim_type(2)");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(0), 0) != mapc[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Au");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(1), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(2), 0) != mapb[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Eg");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Multiplication of a group with one element of Au + Eu symmetry and
		a group with one element of Eu symmetry (3-space), second group
		permuted via [012->120]. The result is expected to contain an element
		of Eu symmetry.
 **/
void so_mult_impl_label_test::test_5b() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_5b()";

	typedef se_label<3, double> se3_t;
	typedef so_mult<3, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se3_t> so_mult_impl_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 8; i3b[1] = 8; i3b[2] = 8;

	block_index_space<3> bis3(dimensions<3>(index_range<3>(i3a, i3b)));

 	mask<3> m3, m3a, m3b, m3c;
 	m3[0] = true; m3[1] = true; m3[2] = true;
 	m3a[0] = true; m3b[1] = true; m3c[2] = true;
	bis3.split(m3, 2); bis3.split(m3, 4); bis3.split(m3, 6);

	se3_t elem3_1(bis3.get_block_index_dims(), k_table_id);
	se3_t elem3_2(bis3.get_block_index_dims(), k_table_id);

	size_t mapb[4], mapc[4];
	mapb[0] = 1; mapb[1] = 3; mapb[2] = 2; mapb[3] = 0;
	mapc[0] = 2; mapc[1] = 0; mapc[2] = 3; mapc[3] = 1;
	for (unsigned int i = 0; i < 4; i++) {
		elem3_1.assign(m3a, i, i);
		elem3_1.assign(m3b, i, mapb[i]);
		elem3_1.assign(m3c, i, mapc[i]);
		elem3_2.assign(m3a, i, mapc[i]);
		elem3_2.assign(m3b, i, i);
		elem3_2.assign(m3c, i, mapb[i]);
	}

	se3_t elem3_ref(elem3_1);
	elem3_1.add_target(2);
	elem3_1.add_target(3);
	elem3_2.add_target(3);
	elem3_ref.add_target(3);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set3(se3_t::k_sym_type);
	symmetry_element_set<3, double> set3_ref(se3_t::k_sym_type);

	set1.insert(elem3_1);
	set2.insert(elem3_2);
	set3_ref.insert(elem3_ref);

	symmetry_operation_params<so_mult_t> params(
			set1, permutation<3>(),
			set2, permutation<3>().permute(0, 1).permute(1, 2), set3);

	so_mult_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis3, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<3, double, se3_t> adapter(set3);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem3_3 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem3_3.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 1");
	}
	if(elem3_3.get_target(0) != 3) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_target(0) != Eu");
	}
	if(elem3_3.get_dim_type(0) == elem3_3.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(1)");
	}
	if(elem3_3.get_dim_type(0) == elem3_3.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(2)");
	}
	if(elem3_3.get_dim_type(1) == elem3_3.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(1) == elem.get_dim_type(2)");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(1), 0) != mapb[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Eg");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(2), 0) != mapc[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Au");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Multiplication of a group with one element of Ag + Au symmetry and
		a group with one element of Ag + Eg symmetry (3-space), first group
		permuted via [012->201], second via [012->120]. The result is expected
		to contain an element of Ag symmetry.
 **/
void so_mult_impl_label_test::test_5c() throw(libtest::test_exception) {

	static const char *testname = "so_mult_impl_label_test::test_5c()";

	typedef se_label<3, double> se3_t;
	typedef so_mult<3, double> so_mult_t;
	typedef symmetry_operation_impl<so_mult_t, se3_t> so_mult_impl_t;

	try {

	index<3> i3a, i3b;
	i3b[0] = 8; i3b[1] = 8; i3b[2] = 8;

	block_index_space<3> bis3(dimensions<3>(index_range<3>(i3a, i3b)));

 	mask<3> m3, m3a, m3b, m3c;
 	m3[0] = true; m3[1] = true; m3[2] = true;
 	m3a[0] = true; m3b[1] = true; m3c[2] = true;
	bis3.split(m3, 2); bis3.split(m3, 4); bis3.split(m3, 6);

	se3_t elem3_1(bis3.get_block_index_dims(), k_table_id);
	se3_t elem3_2(bis3.get_block_index_dims(), k_table_id);
	se3_t elem3_ref(bis3.get_block_index_dims(), k_table_id);

	size_t mapb[4], mapc[4];
	mapb[0] = 1; mapb[1] = 3; mapb[2] = 2; mapb[3] = 0;
	mapc[0] = 2; mapc[1] = 0; mapc[2] = 3; mapc[3] = 1;
	for (unsigned int i = 0; i < 4; i++) {
		elem3_1.assign(m3a, i, i);
		elem3_1.assign(m3b, i, mapb[i]);
		elem3_1.assign(m3c, i, mapc[i]);
		elem3_2.assign(m3a, i, mapb[i]);
		elem3_2.assign(m3b, i, mapc[i]);
		elem3_2.assign(m3c, i, i);
		elem3_ref.assign(m3a, i, mapc[i]);
		elem3_ref.assign(m3b, i, i);
		elem3_ref.assign(m3c, i, mapb[i]);
	}

	elem3_1.add_target(0);
	elem3_1.add_target(2);
	elem3_2.add_target(0);
	elem3_2.add_target(1);
	elem3_ref.add_target(0);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set3(se3_t::k_sym_type);
	symmetry_element_set<3, double> set3_ref(se3_t::k_sym_type);

	set1.insert(elem3_1);
	set2.insert(elem3_2);
	set3_ref.insert(elem3_ref);

	symmetry_operation_params<so_mult_t> params(
			set1, permutation<3>().permute(1, 2).permute(0, 1),
			set2, permutation<3>().permute(0, 1).permute(1, 2), set3);

	so_mult_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis3, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<3, double, se3_t> adapter(set3);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem3_3 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem3_3.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_n_targets() != 1");
	}
	if(elem3_3.get_target(0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_target(0) != Ag");
	}
	if(elem3_3.get_dim_type(0) == elem3_3.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(1)");
	}
	if(elem3_3.get_dim_type(0) == elem3_3.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(0) == elem.get_dim_type(2)");
	}
	if(elem3_3.get_dim_type(1) == elem3_3.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem.get_dim_type(1) == elem.get_dim_type(2)");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(0), 0) != mapc[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Au");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(1), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem3_3.get_label(elem3_3.get_dim_type(2), 0) != mapb[0]) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Eg");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}





} // namespace libtensor
