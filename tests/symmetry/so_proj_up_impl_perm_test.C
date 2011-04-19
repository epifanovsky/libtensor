#include <libtensor/symmetry/so_proj_up_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_proj_up_impl_perm_test.h"

namespace libtensor {


void so_proj_up_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
	test_7();
}


/**	\test Tests that a projection of an empty group yields an empty group
		of a higher order
 **/
void so_proj_up_impl_perm_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_1()";

	typedef se_perm<2, double> se_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<3, double> set2(se_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se_t::k_sym_type);

	mask<3> msk; msk[0] = true; msk[1] = true;
	symmetry_operation_params< so_proj_up<2, 1, double> > params(
		set1, permutation<2>(), msk, bis, set2);

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


/**	\test Projects a group with one element [01->10] from a 2-space to
		a 3-space with a mask [110]. The result is expected to contain
		a single element [012->102]. The permutation is symmetric.
 **/
void so_proj_up_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_2()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<3, double> se3_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se2_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	permutation<2> p10;
	p10.permute(0, 1);
	se_perm<2, double> elem10(p10, true);

	permutation<3> p102;
	p102.permute(0, 1);
	se_perm<3, double> elem102(p102, true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem10);
	set2_ref.insert(elem102);

	mask<3> msk; msk[0] = true; msk[1] = true;
	symmetry_operation_params< so_proj_up<2, 1, double> > params(
		set1, permutation<2>(), msk, bis, set2);

	so_proj_up_impl_t().perform(params);


	compare_ref<3>::compare(testname, bis, set2, set2_ref);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<3> p2; p2.permute(0, 1);
	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(!elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a group with one element [01->10] from a 2-space to
		a 3-space with a mask [101]. The result is expected to contain
		a single element [012->210]. The permutation is symmetric.
 **/
void so_proj_up_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_3()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<3, double> se3_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se2_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	permutation<2> p1; p1.permute(0, 1);
	se_perm<2, double> elem1(p1, true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem1);

	mask<3> msk; msk[0] = true; msk[2] = true;
	symmetry_operation_params< so_proj_up<2, 1, double> > params(
		set1, permutation<2>(), msk, bis, set2);

	so_proj_up_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<3> p2; p2.permute(0, 2);
	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(!elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a group with one element [01->10] from a 2-space to
		a 3-space with a mask [011]. The result is expected to contain
		a single element [012->021]. The permutation is anti-symmetric.
 **/
void so_proj_up_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_4()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<3, double> se3_t;
	typedef so_proj_up<2, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se2_t>
		so_proj_up_impl_t;

	try {

	index<3> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2;
	block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
	mask<3> m;
	m[0] = true; m[1] = true; m[2] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	permutation<2> p1; p1.permute(0, 1);
	se_perm<2, double> elem1(p1, false);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<3, double> set2_ref(se3_t::k_sym_type);

	set1.insert(elem1);

	mask<3> msk; msk[1] = true; msk[2] = true;
	symmetry_operation_params< so_proj_up<2, 1, double> > params(
		set1, permutation<2>(), msk, bis, set2);

	so_proj_up_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<3> p2; p2.permute(1, 2);
	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a group with one element [012->102] from a 3-space to
		a 4-space with a mask [0111] and permutation [012->201].
		The result is expected to contain a single element [0123->0132].
		The permutation is symmetric.
 **/
void so_proj_up_impl_perm_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_5()";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<4, double> se4_t;
	typedef so_proj_up<3, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se3_t>
		so_proj_up_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	permutation<3> p1; p1.permute(0, 1);
	se_perm<3, double> elem1(p1, true);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem1);

	mask<4> msk; msk[1] = true; msk[2] = true; msk[3] = true;
	permutation<3> perm; perm.permute(0, 2).permute(1, 2);
	symmetry_operation_params< so_proj_up<3, 1, double> > params(
		set1, perm, msk, bis, set2);

	so_proj_up_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(2, 3);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(!elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a group with one element [012->102] from a 3-space to
		a 4-space with a mask [1011] and permutation [012->102].
		The result is expected to contain a single element [0123->2103].
		The permutation is anti-symmetric.
 **/
void so_proj_up_impl_perm_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_6()";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<4, double> se4_t;
	typedef so_proj_up<3, 1, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se3_t>
		so_proj_up_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	permutation<3> p1; p1.permute(0, 1);
	se_perm<3, double> elem1(p1, false);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem1);

	mask<4> msk; msk[0] = true; msk[2] = true; msk[3] = true;
	permutation<3> perm; perm.permute(0, 1);
	symmetry_operation_params< so_proj_up<3, 1, double> > params(
		set1, perm, msk, bis, set2);

	so_proj_up_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(0, 2);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projects a group with one cyclic element [0123->1230] from
		a 4-space to another 4-space with a mask [1111] and permutation
		[0123->0231]. The result is expected to contain a single cyclic
		element [0123->2310]. The permutation is symmetric.
 **/
void so_proj_up_impl_perm_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "so_proj_up_impl_perm_test::test_7()";

	typedef se_perm<4, double> se4_t;
	typedef so_proj_up<4, 0, double> so_proj_up_t;
	typedef symmetry_operation_impl<so_proj_up_t, se4_t>
		so_proj_up_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	permutation<4> p1; p1.permute(0, 1).permute(1, 2).permute(2, 3);
	se4_t elem1(p1, true);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem1);

	mask<4> msk; msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	permutation<4> perm; perm.permute(1, 2).permute(2, 3);
	symmetry_operation_params< so_proj_up<4, 0, double> > params(
		set1, perm, msk, bis, set2);

	so_proj_up_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(1, 2).permute(1, 3).permute(0, 3);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(!elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
