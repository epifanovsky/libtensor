#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_apply_impl_label.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_apply_impl_label_test.h"

namespace libtensor {

const char *so_apply_impl_label_test::k_table_id = "point_group";

void so_apply_impl_label_test::perform() throw(libtest::test_exception) {

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
		fail_test("so_apply_impl_perm_test::perform()", __FILE__, __LINE__,
				e.what());
	}

	try {

	test_1(false, false);
	test_1(false, true);
	test_1(true, false);

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(k_table_id);
		throw;
	}

	product_table_container::get_instance().erase(k_table_id);

}


/**	\test Tests application on a group with permutation
 **/
void so_apply_impl_label_test::test_1(
		bool even, bool odd) throw(libtest::test_exception) {

	if (even) odd = false;

	std::ostringstream tnss;
	tnss << "so_apply_impl_label_test::test_1("
			<< (even ? "true" : "false") << ", "
			<< (odd ? "true" : "false") << ")";

	typedef se_label<4, double> se4_t;
	typedef so_apply<4, double> so_t;
	typedef symmetry_operation_impl<so_t, se4_t> so_impl_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m4, m4a, m4b, m4c, m4d;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	m4a[0] = true; m4a[1] = true; m4b[2] = true; m4b[3] = true;
	m4c[0] = true; m4d[1] = true; m4c[2] = true; m4d[3] = true;
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	se4_t elem4a(bis4.get_block_index_dims(), k_table_id);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	bis4.permute(perm);

	se4_t elem4_ref(bis4.get_block_index_dims(), k_table_id);

	for (unsigned int i = 0; i < 4; i++) {
		elem4a.assign(m4a, i, i);
		elem4_ref.assign(m4c, i, i);
	}
	elem4a.assign(m4b, 0, 3); elem4a.assign(m4b, 1, 0);
	elem4a.assign(m4b, 2, 1); elem4a.assign(m4b, 3, 2);
	elem4_ref.assign(m4d, 0, 3); elem4_ref.assign(m4d, 1, 0);
	elem4_ref.assign(m4d, 2, 1); elem4_ref.assign(m4d, 3, 2);

	elem4a.add_target(2);
	elem4_ref.add_target(0);
	elem4_ref.add_target(1);
	elem4_ref.add_target(2);
	elem4_ref.add_target(3);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem4a);
	set2_ref.insert(elem4_ref);

	symmetry_operation_params<so_t> params(set1, perm, even, odd, set2);

	so_impl_t().perform(params);

	compare_ref<4>::compare(tnss.str().c_str(), bis4, set2, set2_ref);

	if(set2.is_empty()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}






} // namespace libtensor
