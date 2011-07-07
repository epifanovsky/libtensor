#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/label/point_group_table.h>
#include <libtensor/symmetry/label/product_table_container.h>
#include <libtensor/symmetry/label/so_permute_impl_label.h>
#include "../compare_ref.h"
#include "so_permute_impl_label_test.h"

namespace libtensor {

const char *so_permute_impl_label_test::k_table_id = "s6";

void so_permute_impl_label_test::perform() throw(libtest::test_exception) {

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
		fail_test("so_permute_impl_perm_test::perform()", __FILE__, __LINE__,
				e.what());
	}

	try {

	test_1();

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(k_table_id);
		throw;
	}

	product_table_container::get_instance().erase(k_table_id);

}


/**	\test Permutes a group with one element of Au symmetry.
 **/
void so_permute_impl_label_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_permute_impl_label_test::test_1()";

	typedef se_label<4, double> se4_t;
	typedef so_permute<4, double> so_permute_t;
	typedef symmetry_operation_impl<so_permute_t, se4_t>
		so_permute_impl_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m4, m4a, m4b, m4c, m4d;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	m4a[0] = true; m4a[1] = true; m4b[2] = true; m4b[3] = true;
	m4c[0] = true; m4d[1] = true; m4c[2] = true; m4d[3] = true;
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se4_t elem4a(bis4.get_block_index_dims());
	{
	    label_set<4> &ss4a = elem4a.create_subset(k_table_id);
	    for (unsigned int i = 0; i < 4; i++) {
	        ss4a.assign(m4a, i, i);
	    }
	    ss4a.assign(m4b, 0, 3);
	    ss4a.assign(m4b, 1, 0);
	    ss4a.assign(m4b, 2, 1);
	    ss4a.assign(m4b, 3, 2);
	    ss4a.add_intrinsic(2);

	}

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	bis4.permute(perm);

	se4_t elem4_ref(bis4.get_block_index_dims());
	{
	    label_set<4> &ss4_ref = elem4_ref.create_subset(k_table_id);

	    for (unsigned int i = 0; i < 4; i++) {
	        ss4_ref.assign(m4c, i, i);
	    }
	    ss4_ref.assign(m4d, 0, 3); ss4_ref.assign(m4d, 1, 0);
	    ss4_ref.assign(m4d, 2, 1); ss4_ref.assign(m4d, 3, 2);

	    ss4_ref.add_intrinsic(2);
	}

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem4a);
	set2_ref.insert(elem4_ref);

	symmetry_operation_params<so_permute_t> params(set1, perm, set2);

	so_permute_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis4, set2, set2_ref);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}






} // namespace libtensor
