#include <libtensor/symmetry/so_permute_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "so_permute_impl_part_test.h"
#include "compare_ref.h"

namespace libtensor {

void so_permute_impl_part_test::perform() throw(libtest::test_exception) {

	test_1();

}


/**	\test Permutes a group with one element of Au symmetry.
 **/
void so_permute_impl_part_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_permute_impl_part_test::test_1()";

	typedef se_part<4, double> se4_t;
	typedef so_permute<4, double> so_permute_t;
	typedef symmetry_operation_impl<so_permute_t, se4_t>
		so_permute_impl_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 7; i4b[1] = 7; i4b[2] = 7; i4b[3] = 7;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m4;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	index<4> i0000, i0011, i1100, i0110, i1001, i0101, i1010, i1111;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	se4_t elem(bis4, m4, 2);
	elem.add_map(i0000, i1111, true);
	elem.add_map(i0011, i1100, true);
	elem.add_map(i0110, i1001, true);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	bis4.permute(perm);

	se4_t elem_ref(bis4, m4, 2);
	elem_ref.add_map(i0000, i1111, true);
	elem_ref.add_map(i0101, i1010, true);
	elem_ref.add_map(i1100, i0011, true);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem_ref);

	symmetry_operation_params<so_permute_t> params(set1, perm, set2);

	so_permute_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}






} // namespace libtensor
