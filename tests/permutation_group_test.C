#include <algorithm>
#include <sstream>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/permutation_group.h>
#include "permutation_group_test.h"

namespace libtensor {


void permutation_group_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2a();
	test_2b();
	test_3();
	test_4();
	test_5();
	test_6a();
	test_6b();
	test_7();
	test_8();

	test_project_down_1();
	test_project_down_2();
	test_project_down_3();
	test_project_down_4();
	test_project_down_8a();
	test_project_down_8b();

	test_permute_1();
	test_permute_2();
	test_permute_3();
}


/**	\test Tests the C1 group in a 4-space
 **/
void permutation_group_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_1()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation_group<4, double> pg;

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S2(+) group in a 2-space
 **/
void permutation_group_test::test_2a() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_2a()";

	typedef se_perm<2, double> se_perm_t;

	try {

	permutation<2> perm1; perm1.permute(0, 1);

	symmetry_element_set<2, double> set1(se_perm_t::k_sym_type);
	set1.insert(se_perm_t(perm1, true));
	permutation_group<2, double> pg(set1);
	
	std::list< permutation<2> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<2>());
	lst_ref_symm.push_back(perm1);
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S2(-) group in a 2-space
 **/
void permutation_group_test::test_2b() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_2b()";

	typedef se_perm<2, double> se_perm_t;

	try {

	permutation<2> perm1; perm1.permute(0, 1);

	symmetry_element_set<2, double> set1(se_perm_t::k_sym_type);
	set1.insert(se_perm_t(perm1, false));
	permutation_group<2, double> pg(set1);
	
	std::list< permutation<2> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<2>());
	lst_ref_asymm.push_back(perm1);
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S3(+) group in a 3-space. The group is created using
		[012->120] and [012->102].
 **/
void permutation_group_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_3()";

	typedef se_perm<3, double> se_perm_t;

	try {

	symmetry_element_set<3, double> set1(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(permutation<3>().permute(0, 1).
		permute(1, 2), true));
	set1.insert(se_perm_t(permutation<3>().permute(0, 1), true));
	permutation_group<3, double> pg(set1);
	
	std::list< permutation<3> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<3>());
	lst_ref_symm.push_back(permutation<3>().permute(0, 1));
	lst_ref_symm.push_back(permutation<3>().permute(0, 2));
	lst_ref_symm.push_back(permutation<3>().permute(1, 2));
	lst_ref_symm.push_back(permutation<3>().permute(0, 1).permute(1, 2));
	lst_ref_symm.push_back(permutation<3>().permute(1, 2).permute(0, 1));
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the A4(+) group in a 4-space
 **/
void permutation_group_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_4()";

	try {

	symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
	set.insert(se_perm<4, double>(permutation<4>().
		permute(0, 1).permute(2, 3), true));
	set.insert(se_perm<4, double>(permutation<4>().
		permute(0, 1).permute(1, 2), true));
	permutation_group<4, double> pg(set);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	lst_ref_symm.push_back(permutation<4>().permute(0, 1).permute(1, 2));
	lst_ref_symm.push_back(permutation<4>().permute(1, 2).permute(0, 1));
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(2, 3).permute(0, 2));
	lst_ref_symm.push_back(permutation<4>().permute(0, 1).permute(1, 3));
	lst_ref_symm.push_back(permutation<4>().permute(1, 3).permute(0, 1));
	lst_ref_symm.push_back(permutation<4>().permute(1, 2).permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(2, 3).permute(1, 2));
	lst_ref_symm.push_back(permutation<4>().permute(0, 1).permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(1, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 3).permute(1, 2));
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S2(+)*S2(+) group in a 4-space
 **/
void permutation_group_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_5()";

	try {

	symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
	set.insert(se_perm<4, double>(permutation<4>().
		permute(0, 1).permute(2, 3), true));
	set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
	permutation_group<4, double> pg(set);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	lst_ref_symm.push_back(permutation<4>().permute(0, 1));
	lst_ref_symm.push_back(permutation<4>().permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 1).permute(2, 3));
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S4(+) group in a 4-space
 **/
void permutation_group_test::test_6a() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_6a()";

	try {

	symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
	set.insert(se_perm<4, double>(permutation<4>().
		permute(0, 1).permute(1, 2).permute(2, 3), true));
	set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
	permutation_group<4, double> pg(set);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	all_permutations(lst_ref_symm);
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S4(-) group in a 4-space
 **/
void permutation_group_test::test_6b() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_6b()";

	try {

	symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
	set.insert(se_perm<4, double>(permutation<4>().
		permute(0, 1).permute(1, 2).permute(2, 3), false));
	set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), false));
	permutation_group<4, double> pg(set);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	all_permutations(lst_ref_asymm);
	lst_ref_asymm.erase(std::find(lst_ref_asymm.begin(),
		lst_ref_asymm.end(), permutation<4>()));
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests a symmetric perm group in a 4-space
 **/
void permutation_group_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_7()";

	typedef se_perm<4, double> se_perm_t;

	try {

	symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
	set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
	set.insert(se_perm<4, double>(permutation<4>().permute(2, 3), true));
	set.insert(se_perm<4, double>(permutation<4>().permute(0, 2).
		permute(1, 3), true));
	permutation_group<4, double> pg(set);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(1, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 1));
	lst_ref_symm.push_back(permutation<4>().permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 1).permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(1, 3).
		permute(0, 1));
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(1, 3).
		permute(2, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 3).permute(1, 2));
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests a symmetric perm group in a 6-space
 **/
void permutation_group_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_8()";

	try {

	symmetry_element_set<6, double> set(se_perm<6, double>::k_sym_type);
	set.insert(se_perm<6, double>(permutation<6>().permute(0, 3).
		permute(1, 4).permute(2, 5), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(1, 2), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(4, 5), true));
	permutation_group<6, double> pg(set);

	std::list< permutation<6> > lst_ref_symm, lst_ref_asymm;
	permutation<6> perm;
	perm.permute(0, 3).permute(1, 4).permute(2, 5);
	lst_ref_symm.push_back(permutation<6>());
	lst_ref_symm.push_back(permutation<6>().permute(1, 2));
	lst_ref_symm.push_back(permutation<6>().permute(4, 5));
	lst_ref_symm.push_back(permutation<6>().permute(1, 2).permute(4, 5));
	lst_ref_symm.push_back(perm);
	lst_ref_symm.push_back(permutation<6>().permute(perm).permute(1, 2));
	lst_ref_symm.push_back(permutation<6>().permute(perm).permute(4, 5));
	lst_ref_symm.push_back(permutation<6>().permute(perm).
		permute(1, 2).permute(4, 5));
	verify_members(testname, pg, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the projection of the S4(+) group in a 4-space onto
		a 2-space, S2(+)
 **/
void permutation_group_test::test_project_down_1()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_1()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2).permute(2, 3);
	permutation<4> perm2; perm2.permute(0, 1);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg4(set1);

	permutation_group<2, double> pg2;
	mask<4> msk; msk[1] = true; msk[3] = true;
	pg4.project_down(msk, pg2);

	permutation<2> p2_1, p2_2; p2_2.permute(0, 1);

	if(!pg2.is_member(true, p2_1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(true, p2_1)");
	}
	if(!pg2.is_member(true, p2_2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(true, p2_2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the projection of the C4(+) group in a 4-space onto
		a 2-space, C1
 **/
void permutation_group_test::test_project_down_2()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_2()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2).permute(2, 3);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
	set1.insert(se_perm_t(perm1, true));
	permutation_group<4, double> pg4(set1);

	permutation_group<2, double> pg2;
	mask<4> msk; msk[0] = true; msk[1] = true;
	pg4.project_down(msk, pg2);

	permutation<2> p2_1, p2_2; p2_2.permute(0, 1);

	if(!pg2.is_member(true, p2_1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(true, p2_1)");
	}
	if(pg2.is_member(true, p2_2)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg2.is_member(true, p2_2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the projection of the S4(-) group in a 4-space onto
		a 2-space, S2(-)
 **/
void permutation_group_test::test_project_down_3()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_3()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2).permute(2, 3);
	permutation<4> perm2; perm2.permute(0, 1);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
	set1.insert(se_perm_t(perm1, false));
	set1.insert(se_perm_t(perm2, false));
	permutation_group<4, double> pg4(set1);

	permutation_group<2, double> pg2;
	mask<4> msk; msk[2] = true; msk[3] = true;
	pg4.project_down(msk, pg2);

	permutation<2> p2_1, p2_2; p2_2.permute(0, 1);

	if(!pg2.is_member(true, p2_1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(true, p2_1)");
	}
	if(!pg2.is_member(false, p2_2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(false, p2_2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests the projection of the S2(-) group in a 2-space onto
		a 1-space
 **/
void permutation_group_test::test_project_down_4()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_4()";

	typedef se_perm<2, double> se_perm_t;

	try {

	bool symm = false;
	permutation<2> perm1; perm1.permute(0, 1);

	symmetry_element_set<2, double> set1(se_perm_t::k_sym_type);
	set1.insert(se_perm_t(perm1, symm));
	permutation_group<2, double> pg2(set1);

	permutation_group<1, double> pg1;
	mask<2> msk; msk[0] = true;
	pg2.project_down(msk, pg1);

	permutation<1> p1;

	if(!pg1.is_member(true, p1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg1.is_member(true, p1)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests a symmetric perm group in a 6-space
	\sa test_8
 **/
void permutation_group_test::test_project_down_8a()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_8a()";

	try {

	symmetry_element_set<6, double> set(se_perm<6, double>::k_sym_type);
	set.insert(se_perm<6, double>(permutation<6>().permute(0, 3).
		permute(1, 4).permute(2, 5), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(1, 2), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(4, 5), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(2, 5), true));
	permutation_group<6, double> pg(set);
	permutation_group<4, double> pg2;
	mask<6> m;
	m[0] = true; m[1] = true; m[3] = true; m[4] = true;
	pg.project_down(m, pg2);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	lst_ref_symm.push_back(permutation<4>().permute(0, 2));
	lst_ref_symm.push_back(permutation<4>().permute(1, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(1, 3));
	verify_members(testname, pg2, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg2, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests a symmetric perm group in a 6-space
	\sa test_8
 **/
void permutation_group_test::test_project_down_8b()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_8b()";

	try {

	symmetry_element_set<6, double> set(se_perm<6, double>::k_sym_type);
	set.insert(se_perm<6, double>(permutation<6>().permute(0, 3).
		permute(1, 4).permute(2, 5), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(1, 2), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(4, 5), true));
	set.insert(se_perm<6, double>(permutation<6>().permute(2, 4), true));
	permutation_group<6, double> pg(set);
	permutation_group<4, double> pg2;
	mask<6> m;
	m[0] = true; m[1] = true; m[3] = true; m[5] = true;
	pg.project_down(m, pg2);

	std::list< permutation<4> > lst_ref_symm, lst_ref_asymm;
	lst_ref_symm.push_back(permutation<4>());
	lst_ref_symm.push_back(permutation<4>().permute(0, 2));
	lst_ref_symm.push_back(permutation<4>().permute(1, 3));
	lst_ref_symm.push_back(permutation<4>().permute(0, 2).permute(1, 3));
	verify_members(testname, pg2, lst_ref_symm, lst_ref_asymm);
	verify_genset(testname, pg2, lst_ref_symm, lst_ref_asymm);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Test the identity %permutation on S2(+)*S2(+).
 **/
void permutation_group_test::test_permute_1() throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_permute_1()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1);
	permutation<4> perm2; perm2.permute(2, 3);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg4(set1);

	permutation<4> perm0;
	pg4.permute(perm0);

	if(!pg4.is_member(true, perm1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm1)");
	}
	if(!pg4.is_member(true, perm2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Test a non-identity %permutation on S2(+)*S2(+).
 **/
void permutation_group_test::test_permute_2() throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_permute_2()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1);
	permutation<4> perm2; perm2.permute(2, 3);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg4(set1);

	permutation<4> perm0;
	perm0.permute(1, 2);
	pg4.permute(perm0);

	perm1.permute(0, 1).permute(0, 2);
	if(!pg4.is_member(true, perm1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm1)");
	}
	perm2.permute(2, 3).permute(1, 3);
	if(!pg4.is_member(true, perm2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Test a non-identity %permutation on S3(+)*C1.
 **/
void permutation_group_test::test_permute_3() throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_permute_3()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2);
	permutation<4> perm2; perm2.permute(0, 1);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg4(set1);

	permutation<4> perm0;
	perm0.permute(3, 2).permute(2, 1).permute(1, 0);
	pg4.permute(perm0);

	perm1.reset();
	perm1.permute(1, 2).permute(2, 3);
	if(!pg4.is_member(true, perm1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm1)");
	}
	perm2.reset();
	perm2.permute(2, 3);
	if(!pg4.is_member(true, perm2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm2)");
	}

	permutation<4> perm3;
	perm3.permute(1, 3);
	if(!pg4.is_member(true, perm3)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg4.is_member(true, perm3)");
	}

	permutation<4> perm4;
	perm4.permute(0, 3);
	if(pg4.is_member(true, perm4)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg4.is_member(true, perm4)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


template<size_t N>
void permutation_group_test::verify_group(const char *testname,
	const std::list< permutation<N> > &lst)
	throw(libtest::test_exception) {

	for(typename std::list< permutation<N> >::const_iterator i =
		lst.begin(); i != lst.end(); i++) {
	for(typename std::list< permutation<N> >::const_iterator j =
		lst.begin(); j != lst.end(); j++) {

		permutation<N> p(*i); p.permute(*j);
		if(std::find(lst.begin(), lst.end(), p) == lst.end()) {
			std::ostringstream ss;
			ss << "Not a group: missing " << p;
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}
	}
}


template<size_t N, typename T>
void permutation_group_test::verify_members(const char *testname,
	const permutation_group<N, T> &grp,
	const std::list< permutation<N> > &allowed_symm,
	const std::list< permutation<N> > &allowed_asymm)
	throw(libtest::test_exception) {

	std::list< permutation<N> > all;
	all_permutations(all);

	for(typename std::list< permutation<N> >::iterator i = all.begin();
		i != all.end(); i++) {

		typename std::list< permutation<N> >::const_iterator isymm =
			std::find(allowed_symm.begin(), allowed_symm.end(), *i);
		typename std::list< permutation<N> >::const_iterator iasymm =
			std::find(allowed_asymm.begin(),
			allowed_asymm.end(), *i);
		bool bsymm = grp.is_member(true, *i);
		bool basymm = grp.is_member(false, *i);
		if(bsymm != (isymm != allowed_symm.end())) {
			std::ostringstream ss;
			ss << "Inconsistent symm: " << *i << " ";
			if(bsymm) ss << "should not be allowed.";
			else ss << "should be allowed.";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
		if(basymm != (iasymm != allowed_asymm.end())) {
			std::ostringstream ss;
			ss << "Inconsistent asymm: " << *i << " ";
			if(basymm) ss << "should not be allowed.";
			else ss << "should be allowed.";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}
}


template<size_t N, typename T>
void permutation_group_test::verify_genset(const char *testname,
	const permutation_group<N, T> &grp,
	const std::list< permutation<N> > &allowed_symm,
	const std::list< permutation<N> > &allowed_asymm)
	throw(libtest::test_exception) {

	symmetry_element_set<N, T> set(se_perm<N, T>::k_sym_type);
	grp.convert(set);
	symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter(set);

	std::list< permutation<N> > lst_symm, lst_asymm;
	gen_group(adapter, true, permutation<N>(), lst_symm);
	gen_group(adapter, false, permutation<N>(), lst_asymm);

	if(lst_symm.size() != allowed_symm.size()) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected group(symm) size");
	}

	for(typename std::list< permutation<N> >::const_iterator i =
		lst_symm.begin(); i != lst_symm.end(); i++) {

		if(std::find(allowed_symm.begin(), allowed_symm.end(), *i) ==
			allowed_symm.end()) {

			std::ostringstream ss;
			ss << "Permutation " << *i << " is not allowed in symm";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	if(lst_asymm.size() != allowed_asymm.size()) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected group(asymm) size");
	}

	for(typename std::list< permutation<N> >::const_iterator i =
		lst_asymm.begin(); i != lst_asymm.end(); i++) {

		if(std::find(allowed_asymm.begin(), allowed_asymm.end(), *i) ==
			allowed_asymm.end()) {

			std::ostringstream ss;
			ss << "Permutation " << *i <<
				" is not allowed in asymm";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}
}


template<size_t N>
void permutation_group_test::all_permutations(
	std::list< permutation<N> > &lst) {

	std::list< permutation<N - 1> > lst2;
	all_permutations(lst2);

	for(typename std::list< permutation<N - 1> >::iterator i2 =
		lst2.begin(); i2 != lst2.end(); i2++) {

		sequence<N - 1, size_t> seq2a(0), seq2b(0);
		for(size_t j = 0; j < N - 1; j++) seq2a[j] = seq2b[j] = j;
		seq2b.permute(*i2);

		for(size_t i = 0; i < N; i++) {

			sequence<N, size_t> seq1a(0), seq1b(0);

			for(size_t j = 0, k = 0; j < N; j++) {
				seq1a[j] = j;
				if(j == i) seq1b[j] = N - 1;
				else { seq1b[j] = seq2b[k]; k++; }
			}

			permutation_builder<N> pb1(seq1b, seq1a);
			lst.push_back(pb1.get_perm());
		}
	}
}


void permutation_group_test::all_permutations(
	std::list< permutation<1> > &lst) {

	lst.push_back(permutation<1>());
}


void permutation_group_test::all_permutations(
	std::list< permutation<0> > &lst) {

}


template<size_t N, typename T>
void permutation_group_test::gen_group(
	const symmetry_element_set_adapter< N, T, se_perm<N, T> > &set,
	bool sign, const permutation<N> &perm0,
	std::list< permutation<N> > &lst) {

	if(std::find(lst.begin(), lst.end(), perm0) != lst.end()) return;
	if(sign || !perm0.is_identity()) lst.push_back(perm0);

	typename symmetry_element_set_adapter< N, T,
		se_perm<N, T> >::iterator i = set.begin();
	for(; i != set.end(); i++) {
		const se_perm<N, T> &e = set.get_elem(i);
		if(e.is_symm() != sign) continue;
		permutation<N> perm1(perm0);
		perm1.permute(e.get_perm());
		gen_group(set, sign, perm1, lst);
	}
}


} // namespace libtensor
