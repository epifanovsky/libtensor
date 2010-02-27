#include <libtensor/symmetry/so_proj_down_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "so_proj_down_impl_perm_test.h"

namespace libtensor {


void so_proj_down_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
	test_7();
}


/**	\test Tests that a projection of an empty group yields an empty group
		of a lower order
 **/
void so_proj_down_impl_perm_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_1()";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<2, double> se2_t;

	try {

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	mask<3> msk; msk[0] = true; msk[1] = true; msk[2] = false;
	symmetry_operation_params< so_proj_down<3, 1, double> >
		params(set1, msk);

	so_proj_down_impl< se_perm<3, double> >().perform(params, set2);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of one 2-cycle in a 2-space onto a 1-space. Expected
		result: C1 in 1-space.
 **/
void so_proj_down_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_2()";

	typedef se_perm<1, double> se1_t;
	typedef se_perm<2, double> se2_t;

	try {

	permutation<2> p1; p1.permute(0, 1);
	se2_t elem1(p1, true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<1, double> set2(se1_t::k_sym_type);

	set1.insert(elem1);

	mask<2> msk; msk[0] = true; msk[1] = false;
	symmetry_operation_params< so_proj_down<2, 1, double> >
		params(set1, msk);

	so_proj_down_impl<se2_t>().perform(params, set2);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of a 2-cycle in a 3-space onto a 2-space untouched by
		the mask. Expected result: 2-cycle in 2-space.
 **/
void so_proj_down_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_3()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<3, double> se3_t;

	try {

	permutation<3> p1; p1.permute(0, 1);
	se3_t elem1(p1, true);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	set1.insert(elem1);

	mask<3> msk; msk[0] = true; msk[1] = true; msk[2] = false;
	symmetry_operation_params< so_proj_down<3, 1, double> >
		params(set1, msk);

	so_proj_down_impl<se3_t>().perform(params, set2);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<2> p2; p2.permute(0, 1);
	symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
	symmetry_element_set_adapter<2, double, se2_t>::iterator i =
		adapter.begin();
	const se2_t &elem2 = adapter.get_elem(i);
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


/**	\test 
 **/
void so_proj_down_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_4()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<3, double> se3_t;

	try {

	//~ permutation<2> p1; p1.permute(0, 1);
	//~ se_perm<2, double> elem1(p1, false);

	//~ symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	//~ symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	//~ set1.insert(elem1);

	//~ mask<3> msk; msk[1] = true; msk[2] = true;
	//~ symmetry_operation_params< so_proj_up<2, 1, double> > params(set1, msk);

	//~ so_proj_up_impl< se_perm<2, double> >().perform(params, set2);

	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected a non-empty set.");
	//~ }

	//~ permutation<3> p2; p2.permute(1, 2);
	//~ symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	//~ symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		//~ adapter.begin();
	//~ const se3_t &elem2 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }
	//~ if(elem2.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "elem2.is_symm()");
	//~ }
	//~ if(!elem2.get_perm().equals(p2)) {
		//~ fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test 
 **/
void so_proj_down_impl_perm_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_5()";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<4, double> se4_t;

	try {

	//~ permutation<3> p1; p1.permute(0, 1);
	//~ se_perm<3, double> elem1(p1, true);

	//~ symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	//~ symmetry_element_set<4, double> set2(se4_t::k_sym_type);

	//~ set1.insert(elem1);

	//~ mask<4> msk; msk[1] = true; msk[2] = true; msk[3] = true;
	//~ permutation<3> perm; perm.permute(0, 2).permute(1, 2);
	//~ symmetry_operation_params< so_proj_up<3, 1, double> > params(
		//~ set1, msk, perm);

	//~ so_proj_up_impl< se_perm<3, double> >().perform(params, set2);

	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected a non-empty set.");
	//~ }

	//~ permutation<4> p2; p2.permute(2, 3);
	//~ symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
	//~ symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		//~ adapter.begin();
	//~ const se4_t &elem2 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }
	//~ if(!elem2.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	//~ }
	//~ if(!elem2.get_perm().equals(p2)) {
		//~ fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test 
 **/
void so_proj_down_impl_perm_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_6()";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<4, double> se4_t;

	try {

	//~ permutation<3> p1; p1.permute(0, 1);
	//~ se_perm<3, double> elem1(p1, false);

	//~ symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	//~ symmetry_element_set<4, double> set2(se4_t::k_sym_type);

	//~ set1.insert(elem1);

	//~ mask<4> msk; msk[0] = true; msk[2] = true; msk[3] = true;
	//~ permutation<3> perm; perm.permute(0, 1);
	//~ symmetry_operation_params< so_proj_up<3, 1, double> > params(
		//~ set1, msk, perm);

	//~ so_proj_up_impl< se_perm<3, double> >().perform(params, set2);

	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected a non-empty set.");
	//~ }

	//~ permutation<4> p2; p2.permute(0, 2);
	//~ symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
	//~ symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		//~ adapter.begin();
	//~ const se4_t &elem2 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }
	//~ if(elem2.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "elem2.is_symm()");
	//~ }
	//~ if(!elem2.get_perm().equals(p2)) {
		//~ fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test 
 **/
void so_proj_down_impl_perm_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "so_proj_down_impl_perm_test::test_7()";

	typedef se_perm<4, double> se4_t;

	try {

	//~ permutation<4> p1; p1.permute(0, 1).permute(1, 2).permute(2, 3);
	//~ se4_t elem1(p1, true);

	//~ symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	//~ symmetry_element_set<4, double> set2(se4_t::k_sym_type);

	//~ set1.insert(elem1);

	//~ mask<4> msk; msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	//~ permutation<4> perm; perm.permute(1, 2).permute(2, 3);
	//~ symmetry_operation_params< so_proj_up<4, 0, double> > params(
		//~ set1, msk, perm);

	//~ so_proj_up_impl< se_perm<4, double> >().perform(params, set2);

	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected a non-empty set.");
	//~ }

	//~ permutation<4> p2; p2.permute(1, 2).permute(1, 3).permute(0, 3);
	//~ symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
	//~ symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		//~ adapter.begin();
	//~ const se4_t &elem2 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }
	//~ if(!elem2.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	//~ }
	//~ if(!elem2.get_perm().equals(p2)) {
		//~ fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
