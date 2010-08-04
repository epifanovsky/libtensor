#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_concat_impl_label.h>
#include <libtensor/btod/transf_double.h>
#include "so_concat_impl_label_test.h"
#include "compare_ref.h"

namespace libtensor {

const char *so_concat_impl_label_test::k_table_id = "point_group";

void so_concat_impl_label_test::perform() throw(libtest::test_exception) {

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
		fail_test("so_concat_impl_perm_test::perform()", __FILE__, __LINE__,
				e.what());
	}

	try {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(k_table_id);
		throw;
	}

	product_table_container::get_instance().erase(k_table_id);

}


/**	\test Tests that a concatenation of two empty group yields an empty group
		of a higher order
 **/
void so_concat_impl_label_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_label_test::test_1()";

	typedef se_label<2, double> se2_t;
	typedef se_label<3, double> se3_t;
	typedef so_concat<2, 3, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<5> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
	block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
	mask<5> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<5, double> set3(se2_t::k_sym_type);
	symmetry_element_set<5, double> set3_ref(se2_t::k_sym_type);

	symmetry_operation_params< so_concat<2, 3, double> > params(
		set1, set2, permutation<5>(), bis, set3);

	so_concat_impl_t().perform(params);

	compare_ref<5>::compare(testname, bis, set3, set3_ref);

	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a group with one element of Au symmetry and an empty
 	 	group (2-space) forming a 4-space. The result is expected to contain
		a single element of Au symmetry with labels assigned to only half of
		the space.
 **/
void so_concat_impl_label_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_label_test::test_2()";

	typedef se_label<2, double> se2_t;
	typedef se_label<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;
	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	mask<4> m4, m;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	m[0] = true; m[1] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se2_t elem2(bis2.get_block_index_dims(), k_table_id);
	se4_t elem4_ref(bis4.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m2, i, i);
	for (unsigned int i = 0; i < 4; i++) elem4_ref.assign(m, i, i);
	elem2.add_target(2);
	elem4_ref.add_target(2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem2);
	set3_ref.insert(elem4_ref);

	symmetry_operation_params< so_concat<2, 2, double> > params(
		set1, set2, permutation<4>(), bis4, set3);

	so_concat_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem4 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem4.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__, "elem4.get_n_targets() != 1");
	}
	if(elem4.get_target(0) != 2) {
		fail_test(testname, __FILE__, __LINE__, "elem4.get_target(0) != Au");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(1)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(2)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(3)");
	}
	if(elem4.get_dim_type(2) != elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(2) != elem4.get_dim_type(3)");
	}
	if(elem4.get_label(elem4.get_dim_type(0), 0) != 0)  {
		fail_test(testname, __FILE__, __LINE__, "elem4.get_label(0, 0) != Ag");
	}
	if(elem4.get_label(elem4.get_dim_type(2), 0) != point_group_table::k_invalid) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(2, 0) not invalid");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a group with one element of Au symmetry and an empty
 	 	group (2-space) forming a 4-space with permutation [0123->1203]. The
 	 	result is expected to contain a single element of Au symmetry with
 	 	labels assigned to only half of the space.
 **/
void so_concat_impl_label_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_label_test::test_3()";

	typedef se_label<2, double> se2_t;
	typedef se_label<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;
	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	mask<4> m4, m;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	m[0] = true; m[2] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se2_t elem2(bis2.get_block_index_dims(), k_table_id);
	se4_t elem4_ref(bis4.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m2, i, i);
	for (unsigned int i = 0; i < 4; i++) elem4_ref.assign(m, i, i);
	elem2.add_target(2);
	elem4_ref.add_target(2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem2);
	set3_ref.insert(elem4_ref);

	symmetry_operation_params< so_concat<2, 2, double> > params(set1,
			set2, permutation<4>().permute(0, 1).permute(1, 2), bis4, set3);

	so_concat_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem4 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem4.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_n_targets() != 1");
	}
	if(elem4.get_target(0) != 2) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_target(0) != Au");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(2)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(1)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(3)");
	}
	if(elem4.get_dim_type(1) != elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(2) != elem4.get_dim_type(3)");
	}
	if(elem4.get_label(elem4.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem4.get_label(elem4.get_dim_type(1), 0) != point_group_table::k_invalid) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(1, 0) not invalid");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Concatenates an empty group (2-space) and a group with one element
		of Au symmetry forming a 4-space with permutation [0123->3021]. The
 	 	result is expected to contain a single element of Au symmetry with
 	 	labels assigned to only half of the space.
 **/
void so_concat_impl_label_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_label_test::test_4()";

	typedef se_label<2, double> se2_t;
	typedef se_label<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;
	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	mask<4> m4, m;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	m[0] = true; m[2] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se2_t elem2(bis2.get_block_index_dims(), k_table_id);
	se4_t elem4_ref(bis4.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) elem2.assign(m2, i, i);
	for (unsigned int i = 0; i < 4; i++) elem4_ref.assign(m, i, i);
	elem2.add_target(2);
	elem4_ref.add_target(2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set2.insert(elem2);
	set3_ref.insert(elem4_ref);

	symmetry_operation_params< so_concat<2, 2, double> > params(set1,
			set2, permutation<4>().permute(0, 3).permute(1, 3), bis4, set3);

	so_concat_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem4 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem4.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_n_targets() != 1");
	}
	if(elem4.get_target(0) != 2) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_target(0) != Au");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(2)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(1)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(3)");
	}
	if(elem4.get_dim_type(1) != elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(2) != elem4.get_dim_type(3)");
	}
	if(elem4.get_label(elem4.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem4.get_label(elem4.get_dim_type(1), 0) != point_group_table::k_invalid) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(1, 0) not invalid");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a group with one element of Au symmetry and a group with
		one element of Eg symmetry forming a 4-space with permutation
		[0123->1203]. The result is expected to contain a single element of Eu
		symmetry.
 **/
void so_concat_impl_label_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_label_test::test_5()";

	typedef se_label<2, double> se2_t;
	typedef se_label<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;
	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	mask<4> m4;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se2_t elem2a(bis2.get_block_index_dims(), k_table_id);
	for (unsigned int i = 0; i < 4; i++) elem2a.assign(m2, i, i);
	se2_t elem2b(elem2a);
	elem2a.add_target(2);
	elem2b.add_target(1);

	se4_t elem4_ref(bis4.get_block_index_dims(), k_table_id);
	for (unsigned int i = 0; i < 4; i++) elem4_ref.assign(m4, i, i);
	elem4_ref.add_target(3);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem2a);
	set2.insert(elem2b);
	set3_ref.insert(elem4_ref);

	symmetry_operation_params< so_concat<2, 2, double> > params(set1,
			set2, permutation<4>().permute(0, 1).permute(1, 2), bis4, set3);

	so_concat_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem4 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem4.get_n_targets() != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_n_targets() != 1");
	}
	if(elem4.get_target(0) != 3) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_target(0) != Eu");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(1)");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(2)");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(3)");
	}
	if(elem4.get_label(elem4.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a group with one element of Au + Eu symmetry and
		a group with one element of Eg symmetry forming a 4-space with
		permutation [0123->1203]. The result is expected to contain a single
		element of Au + Eu symmetry. The subspace are different.
 **/
void so_concat_impl_label_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_label_test::test_6()";

	typedef se_label<2, double> se2_t;
	typedef se_label<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<2> i2a, i2b;
	i2b[0] = 8; i2b[1] = 8;
	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<2> bis2(dimensions<2>(index_range<2>(i2a, i2b)));
	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

 	mask<2> m2;
 	m2[0] = true; m2[1] = true;
	mask<4> m4;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis2.split(m2, 2); bis2.split(m2, 4); bis2.split(m2, 6);
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se2_t elem2a(bis2.get_block_index_dims(), k_table_id);
	se2_t elem2b(bis2.get_block_index_dims(), k_table_id);
	for (unsigned int i = 0; i < 4; i++) elem2a.assign(m2, i, i);
	elem2b.assign(m2, 0, 1);
	elem2b.assign(m2, 1, 3);
	elem2b.assign(m2, 2, 0);
	elem2b.assign(m2, 3, 2);

	elem2a.add_target(2);
	elem2a.add_target(3);
	elem2b.add_target(1);

	se4_t elem4_ref(bis4.get_block_index_dims(), k_table_id);
	mask<4> m4a, m4b;
	m4a[0] = true; m4a[2] = true; m4b[1] = true; m4b[3] = true;
	for (unsigned int i = 0; i < 4; i++) elem4_ref.assign(m4a, i, i);
	elem4_ref.assign(m4b, 0, 1);
	elem4_ref.assign(m4b, 1, 3);
	elem4_ref.assign(m4b, 2, 0);
	elem4_ref.assign(m4b, 3, 2);

	elem4_ref.add_target(2);
	elem4_ref.add_target(3);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem2a);
	set2.insert(elem2b);
	set3_ref.insert(elem4_ref);

	symmetry_operation_params< so_concat<2, 2, double> > params(set1,
			set2, permutation<4>().permute(0, 1).permute(1, 2), bis4, set3);

	so_concat_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis4, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem4 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem4.get_n_targets() != 2) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_n_targets() != 2");
	}
	if(elem4.get_target(0) != 2) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_target(0) != Au");
	}
	if(elem4.get_target(1) != 3) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_target(1) != Eu");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(1)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(1)");
	}
	if(elem4.get_dim_type(0) != elem4.get_dim_type(2)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) != elem4.get_dim_type(2)");
	}
	if(elem4.get_dim_type(0) == elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(0) == elem4.get_dim_type(3)");
	}
	if(elem4.get_dim_type(1) != elem4.get_dim_type(3)) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_dim_type(1) != elem4.get_dim_type(3)");
	}
	if(elem4.get_label(elem4.get_dim_type(0), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(0, 0) != Ag");
	}
	if(elem4.get_label(elem4.get_dim_type(1), 0) != 1) {
		fail_test(testname, __FILE__, __LINE__,
				"elem4.get_label(1, 0) != Eg");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}





} // namespace libtensor
