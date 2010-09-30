#include <libtensor/symmetry/so_proj_up_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include <libtensor/core/block_index_subspace_builder.h>
#include "so_proj_up_impl_part_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_proj_up_impl_part_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2(true);
	test_2(false);
	test_3(true);
	test_3(false);
	test_4(true);
	test_4(false);
	test_5(true);
	test_5(false);
}


/**	\test Tests that a projection of an empty partition set yields an empty
		partition set of a higher order
 **/
void so_proj_up_impl_part_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_part_test::test_1()";

	typedef se_part<2, double> se_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m111, m110;
	m111[0] = true; m111[1] = true; m111[2] = true;
	m110[0] = true; m110[1] = true;
	bis.split(m111, 2);
	bis.split(m111, 3);
	bis.split(m111, 5);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<3, double> set2(se_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se_t::k_sym_type);

	symmetry_operation_params<so_proj_up_t> params(
		set1, permutation<2>(), m110, bis, set2);

	so_proj_up_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis, set2, set2_ref);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a set with one partition from a 2-space to
		a 3-space with a mask [110].
 **/
void so_proj_up_impl_part_test::test_2(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_part_test::test_2(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<3, double> se3_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se2_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m110, m111;
	m110[0] = true; m110[1] = true;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 3);
	bis.split(m111, 5);

	block_index_subspace_builder<2, 1> bb(bis, m110);

	mask<2> m11;
	m11[0] = true; m11[1] = true;
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	index<3> i000, i010, i100, i110;
	i100[0] = 1; i010[1] = 1;
	i110[0] = 1; i110[1] = 1;

	se_part<2, double> elem(bb.get_bis(), m11, 2);
	elem.add_map(i00, i11, sign);
	elem.add_map(i01, i10, !sign);
	se_part<3, double> elem_ref(bis, m110, 2);
	elem_ref.add_map(i000, i110, sign);
	elem_ref.add_map(i010, i100, !sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem_ref);

	symmetry_operation_params<so_proj_up_t> params(
		set1, permutation<2>(), m110, bis, set2);

	so_proj_up_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis, set2, set2_ref);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a set with one partition from a 2-space to
		a 3-space with a mask [101].
 **/
void so_proj_up_impl_part_test::test_3(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_part_test::test_3(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<3, double> se3_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se2_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m101, m111;
	m101[0] = true; m101[2] = true;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 3);
	bis.split(m111, 5);

	block_index_subspace_builder<2, 1> bb(bis, m101);

	mask<2> m11;
	m11[0] = true; m11[1] = true;
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	index<3> i000, i001, i100, i101;
	i100[0] = 1; i001[2] = 1;
	i101[0] = 1; i101[2] = 1;

	se_part<2, double> elem(bb.get_bis(), m11, 2);
	elem.add_map(i00, i11, sign);
	elem.add_map(i01, i10, !sign);
	se_part<3, double> elem_ref(bis, m101, 2);
	elem_ref.add_map(i000, i101, sign);
	elem_ref.add_map(i001, i100, !sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem_ref);

	symmetry_operation_params<so_proj_up_t> params(
		set1, permutation<2>(), m101, bis, set2);

	so_proj_up_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis, set2, set2_ref);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a set with one partition from a 2-space to
		a 3-space with a mask [011].
 **/
void so_proj_up_impl_part_test::test_4(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_part_test::test_4(bool)";

	typedef se_part<2, double> se2_t;
	typedef se_part<3, double> se3_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se2_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m011, m111;
	m011[1] = true; m011[2] = true;
	m111[0] = true; m111[1] = true; m111[2] = true;
	bis.split(m111, 2);
	bis.split(m111, 3);
	bis.split(m111, 5);

	block_index_subspace_builder<2, 1> bb(bis, m011);

	mask<2> m11;
	m11[0] = true; m11[1] = true;
	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;

	index<3> i000, i001, i010, i011;
	i010[1] = 1; i001[2] = 1;
	i011[1] = 1; i011[2] = 1;

	se_part<2, double> elem(bb.get_bis(), m11, 2);
	elem.add_map(i00, i11, sign);
	elem.add_map(i01, i10, !sign);
	se_part<3, double> elem_ref(bis, m011, 2);
	elem_ref.add_map(i000, i011, sign);
	elem_ref.add_map(i001, i010, !sign);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem_ref);

	symmetry_operation_params<so_proj_up_t> params(
		set1, permutation<2>(), m011, bis, set2);

	so_proj_up_impl_t().perform(params);

	compare_ref<3>::compare(testname, bis, set2, set2_ref);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a group with one partition from a 3-space to
		a 4-space with a mask [0111] and permutation [012->201].
 **/
void so_proj_up_impl_part_test::test_5(bool sign)
		throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_part_test::test_5(bool)";

	typedef se_part<3, double> se3_t;
	typedef se_part<4, double> se4_t;
	typedef so_proj_up<3, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se3_t>
		so_proj_up_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 5;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1111, m0111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	m0111[1] = true; m0111[2] = true; m0111[3] = true;
	bis.split(m1111, 2);
	bis.split(m1111, 3);
	bis.split(m1111, 5);

	block_index_subspace_builder<3, 1> bb(bis, m0111);

	mask<3> m111;
	m111[0] = true; m111[1] = true; m111[2] = true;
	index<3> i000, i010, i101, i111;
	i101[0] = 1; i010[1] = 1; i101[2] = 1;
	i111[0] = 1; i111[1] = 1; i111[2] = 1;

	index<4> i0000, i0001, i0110, i0111;
	i0110[1] = 1; i0110[2] = 1; i0001[3] = 1;
	i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;

	se_part<3, double> elem(bb.get_bis(), m111, 2);
	elem.add_map(i000, i111, sign);
	elem.add_map(i010, i101, !sign);
	se_part<4, double> elem_ref(bis, m0111, 2);
	elem_ref.add_map(i0000, i0111, sign);
	elem_ref.add_map(i0001, i0110, !sign);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem_ref);

	permutation<3> perm; perm.permute(0, 2).permute(1, 2);
	symmetry_operation_params< so_proj_up<3, 1, double> > params(
		set1, perm, m0111, bis, set2);

	so_proj_up_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis, set2, set2_ref);


	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
