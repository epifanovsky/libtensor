#include <algorithm>
#include <sstream>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/perm/permutation_group.h>
#include "permutation_group_test.h"

namespace libtensor {


void permutation_group_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2a();
    test_2b();
    test_3();
    test_4();
    test_5a();
    test_5b();
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

    test_stabilize_1();
    test_stabilize_2();
    test_stabilize_3();
    test_stabilize_4();
    test_stabilize_5();
    test_stabilize_6();
    //	test_stabilize_7();

    test_stabilize2_1();
    test_stabilize2_2();

    test_permute_1();
    test_permute_2();
    test_permute_3();
}


/**	\test Tests the C1 group in a 4-space
 **/
void permutation_group_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_1()";

    typedef se_perm<4, double> se_perm_t;
    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        permutation_group<4, double> pg;

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<4>(), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the S2(+) group in a 2-space
 **/
void permutation_group_test::test_2a() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_2a()";

    typedef se_perm<2, double> se_perm_t;
    typedef std::pair<permutation<2>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        permutation<2> perm1; perm1.permute(0, 1);

        symmetry_element_set<2, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(perm1, true));
        permutation_group<2, double> pg(set1);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<2>(), true));
        lst_ref.push_back(signed_perm_t(perm1, true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the S2(-) group in a 2-space
 **/
void permutation_group_test::test_2b() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_2b()";

    typedef se_perm<2, double> se_perm_t;
    typedef std::pair<permutation<2>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        permutation<2> perm1; perm1.permute(0, 1);

        symmetry_element_set<2, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(perm1, false));
        permutation_group<2, double> pg(set1);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<2>(), true));
        lst_ref.push_back(signed_perm_t(perm1, false));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

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
    typedef std::pair<permutation<3>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<3, double> set1(se_perm_t::k_sym_type);

        set1.insert(se_perm_t(permutation<3>().permute(0, 1).
                permute(1, 2), true));
        set1.insert(se_perm_t(permutation<3>().permute(0, 1), true));
        permutation_group<3, double> pg(set1);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<3>(), true));
        lst_ref.push_back(signed_perm_t(permutation<3>().permute(0, 1), true));
        lst_ref.push_back(signed_perm_t(permutation<3>().permute(0, 2), true));
        lst_ref.push_back(signed_perm_t(permutation<3>().permute(1, 2), true));
        lst_ref.push_back(
                signed_perm_t(permutation<3>().permute(0, 1).permute(1, 2), true));
        lst_ref.push_back(
                signed_perm_t(permutation<3>().permute(1, 2).permute(0, 1), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the A4(+) group in a 4-space
 **/
void permutation_group_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_4()";

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
        set.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(2, 3), true));
        set.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(1, 2), true));
        permutation_group<4, double> pg(set);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<4>(), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 1).permute(1, 2), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(1, 2).permute(0, 1), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 2).permute(2, 3), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(2, 3).permute(0, 2), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 1).permute(1, 3), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(1, 3).permute(0, 1), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(1, 2).permute(2, 3), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(2, 3).permute(1, 2), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 1).permute(2, 3), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 2).permute(1, 3), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 3).permute(1, 2), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the S2(+)*S2(+) group in a 4-space
 **/
void permutation_group_test::test_5a() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_5a()";

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
        set.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(2, 3), true));
        set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
        permutation_group<4, double> pg(set);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<4>(), true));
        lst_ref.push_back(signed_perm_t(permutation<4>().permute(0, 1), true));
        lst_ref.push_back(signed_perm_t(permutation<4>().permute(2, 3), true));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 1).permute(2, 3), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Tests the S2(-)*S2(-) group in a 4-space
 **/
void permutation_group_test::test_5b() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_5b()";

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
        set.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(2, 3), true));
        set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), false));
        permutation_group<4, double> pg(set);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(permutation<4>(), true));
        lst_ref.push_back(signed_perm_t(permutation<4>().permute(0, 1), false));
        lst_ref.push_back(signed_perm_t(permutation<4>().permute(2, 3), false));
        lst_ref.push_back(
                signed_perm_t(permutation<4>().permute(0, 1).permute(2, 3), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Tests the S4(+) group in a 4-space
 **/
void permutation_group_test::test_6a() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_6a()";

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
        set.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(1, 2).permute(2, 3), true));
        set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
        permutation_group<4, double> pg(set);

        perm_list_t lst_ref;
        all_permutations(true, lst_ref);
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the S4(-) group in a 4-space
 **/
void permutation_group_test::test_6b() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_6b()";

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
        set.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(1, 2).permute(2, 3), false));
        set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), false));
        permutation_group<4, double> pg(set);

        perm_list_t lst_ref;
        all_permutations(false, lst_ref);
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests a symmetric perm group in a 4-space
 **/
void permutation_group_test::test_7() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_7()";

    typedef se_perm<4, double> se_perm_t;
    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
        set.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
        set.insert(se_perm<4, double>(permutation<4>().permute(2, 3), true));
        set.insert(se_perm<4, double>(permutation<4>().permute(0, 2).
                permute(1, 3), true));
        permutation_group<4, double> pg(set);

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(
                permutation<4>(), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2).permute(1, 3), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 1), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(2, 3), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 1).permute(2, 3), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2).permute(1, 3).permute(0, 1), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2).permute(1, 3).permute(2, 3), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 3).permute(1, 2), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests a symmetric perm group in a 6-space
 **/
void permutation_group_test::test_8() throw(libtest::test_exception) {

    static const char *testname = "permutation_group_test::test_8()";

    typedef std::pair<permutation<6>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

    try {

        symmetry_element_set<6, double> set(se_perm<6, double>::k_sym_type);
        set.insert(se_perm<6, double>(permutation<6>().permute(0, 3).
                permute(1, 4).permute(2, 5), true));
        set.insert(se_perm<6, double>(permutation<6>().permute(1, 2), true));
        set.insert(se_perm<6, double>(permutation<6>().permute(4, 5), true));
        permutation_group<6, double> pg(set);

        perm_list_t lst_ref;
        permutation<6> perm;
        perm.permute(0, 3).permute(1, 4).permute(2, 5);
        lst_ref.push_back(signed_perm_t(permutation<6>(), true));
        lst_ref.push_back(signed_perm_t(permutation<6>().permute(1, 2), true));
        lst_ref.push_back(signed_perm_t(permutation<6>().permute(4, 5), true));
        lst_ref.push_back(signed_perm_t(
                permutation<6>().permute(1, 2).permute(4, 5), true));
        lst_ref.push_back(signed_perm_t(perm, true));
        lst_ref.push_back(signed_perm_t(
                permutation<6>().permute(perm).permute(1, 2), true));
        lst_ref.push_back(signed_perm_t(
                permutation<6>().permute(perm).permute(4, 5), true));
        lst_ref.push_back(signed_perm_t(
                permutation<6>().permute(perm).permute(1, 2).permute(4, 5), true));
        verify_members(testname, pg, lst_ref);
        verify_genset(testname, pg, lst_ref);

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

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

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

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(
                permutation<4>(), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(1, 3), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2).permute(1, 3), true));
        verify_members(testname, pg2, lst_ref);
        verify_genset(testname, pg2, lst_ref);

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

    typedef std::pair<permutation<4>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;

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

        perm_list_t lst_ref;
        lst_ref.push_back(signed_perm_t(
                permutation<4>(), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(1, 3), true));
        lst_ref.push_back(signed_perm_t(
                permutation<4>().permute(0, 2).permute(1, 3), true));
        verify_members(testname, pg2, lst_ref);
        verify_genset(testname, pg2, lst_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {1, 3} in group S6 returning S4.
 **/
void permutation_group_test::test_stabilize_1()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_1()";

    typedef se_perm<6, double> se_perm_t;

    try {

        bool symm = true;

        permutation<6> p1, p2;
        p1.permute(0, 1).permute(1, 2).permute(2, 3).permute(3, 4).permute(4, 5);
        p2.permute(0, 1);
        symmetry_element_set<6, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, symm));
        set1.insert(se_perm_t(p2, symm));

        permutation_group<6, double> pg1(set1), pg2;
        mask<6> msk; msk[1] = true; msk[3] = true;
        pg1.stabilize(msk, pg2);

        permutation<6> p1b, p2b, p3b;
        p1b.permute(0, 2).permute(2, 4).permute(4, 5);
        p2b.permute(0, 2);
        p3b.permute(2, 4).permute(4, 5);

        if(!pg2.is_member(true, p1b)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p1b)");
        }
        if(!pg2.is_member(true, p2b)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p2b)");
        }
        if(!pg2.is_member(true, p3b)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p3b)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {1,3} in group [ijkl] = [klij]
 **/
void permutation_group_test::test_stabilize_2()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_2()";

    typedef se_perm<4, double> se_perm_t;

    try {

        bool symm = true;

        permutation<4> p1; p1.permute(0, 2).permute(1, 3);
        symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, symm));

        permutation_group<4, double> pg1(set1), pg2;

        mask<4> msk; msk[1] = true; msk[3] = true;
        pg1.stabilize(msk, pg2);


        if(!pg2.is_member(true, p1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p1)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {1,2} in group S2 x S2 with
 	 	 pairwise permutation
 **/
void permutation_group_test::test_stabilize_3()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_3()";

    typedef se_perm<4, double> se_perm_t;

    try {

        bool symm = true;

        permutation<4> p1, p2, p3;
        p1.permute(0, 1);
        p2.permute(2, 3);
        p3.permute(0, 2).permute(1, 3);
        symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, symm));
        set1.insert(se_perm_t(p2, symm));
        set1.insert(se_perm_t(p3, true));

        permutation_group<4, double> pg1(set1), pg2;

        mask<4> msk; msk[1] = true; msk[2] = true;
        pg1.stabilize(msk, pg2);

        permutation<4> p1b;
        p1b.permute(0, 3).permute(1, 2);

        if(! pg2.is_member(true, p1b)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p1b)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {0,2} in group A2 x A2 with
 	 	 pairwise permutation
 **/
void permutation_group_test::test_stabilize_4()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_4()";

    typedef se_perm<4, double> se_perm_t;

    try {

        permutation<4> p1, p2, p3;
        p1.permute(0, 1);
        p2.permute(2, 3);
        p3.permute(0, 2).permute(1, 3);
        symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, false));
        set1.insert(se_perm_t(p2, false));
        set1.insert(se_perm_t(p3, true));
        permutation_group<4, double> pg1(set1), pg2;

        mask<4> msk; msk[0] = true; msk[2] = true;
        pg1.stabilize(msk, pg2);

        if(!pg2.is_member(true, p3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p3)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {0,1} in group A2 x A2 with
 	 	 pairwise permutation
 **/
void permutation_group_test::test_stabilize_5()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_5()";

    typedef se_perm<4, double> se_perm_t;

    try {

        permutation<4> p1, p2, p3;
        p1.permute(0, 1);
        p2.permute(2, 3);
        p3.permute(0, 2).permute(1, 3);
        symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, false));
        set1.insert(se_perm_t(p2, false));
        set1.insert(se_perm_t(p3, true));
        permutation_group<4, double> pg1(set1), pg2;

        mask<4> msk; msk[0] = true; msk[1] = true;

        pg1.stabilize(msk, pg2);

        if(!pg2.is_member(false, p1)) {
            fail_test(testname, __FILE__, __LINE__, "!pg2.is_member(true, p1)");
        }
        if(!pg2.is_member(false, p2)) {
            fail_test(testname, __FILE__, __LINE__, "!pg2.is_member(true, p2)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {2, 3, 4} in group S5
 **/
void permutation_group_test::test_stabilize_6()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_6()";

    typedef se_perm<5, double> se_perm_t;

    try {

        symmetry_element_set<5, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(permutation<5>().permute(0, 1).permute(1, 2)
                .permute(2, 3).permute(3, 4), true));
        set1.insert(se_perm_t(permutation<5>().permute(3, 4), true));
        permutation_group<5, double> pg1(set1), pg2;

        mask<5> msk; msk[2] = true; msk[3] = true; msk[4] = true;
        pg1.stabilize(msk, pg2);

        permutation<5> p1;
        p1.permute(0, 1);

        if(!pg2.is_member(true, p1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p1)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Stabilize element set {0,3} in group A2 x A2 with
 	 	 pairwise permutation
 **/
void permutation_group_test::test_stabilize_7()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize_7()";

    typedef se_perm<4, double> se_perm_t;

    try {

        permutation<4> p1, p2, p3;
        p1.permute(0, 1);
        p2.permute(2, 3);
        p3.permute(0, 2).permute(1, 3);
        symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, false));
        set1.insert(se_perm_t(p2, false));
        set1.insert(se_perm_t(p3, true));
        permutation_group<4, double> pg1(set1), pg2;

        mask<4> msk; msk[0] = true; msk[3] = true;
        pg1.stabilize(msk, pg2);

        permutation<4> p1b;
        p1b.permute(0, 3).permute(1, 2);

        if(!pg2.is_member(true, p1b)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p1b)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void permutation_group_test::test_stabilize2_1()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize2_1()";

    typedef se_perm<6, double> se_perm_t;

    try {

        permutation<6> p1, p2, p3, p4, p5;
        p1.permute(0, 1);
        p2.permute(1, 2);
        p3.permute(3, 4);
        p4.permute(4, 5);
        p5.permute(0, 3).permute(1, 4).permute(2, 5);
        symmetry_element_set<6, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, true));
        set1.insert(se_perm_t(p2, true));
        set1.insert(se_perm_t(p3, true));
        set1.insert(se_perm_t(p4, true));
        set1.insert(se_perm_t(p5, true));
        permutation_group<6, double> pg1(set1), pg2;

        mask<6> msk[2];
        msk[0][2] = true; msk[0][5] = true;
        msk[1][1] = true; msk[1][4] = true;
        pg1.stabilize(msk, pg2);

        if(!pg2.is_member(true, p5)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg2.is_member(true, p5)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void permutation_group_test::test_stabilize2_2()
throw(libtest::test_exception) {

    static const char *testname =
            "permutation_group_test::test_stabilize2_2()";

    typedef se_perm<8, double> se_perm_t;

    try {

        permutation<8> p1, p2, p3, p4, p5, p6, p7;
        p1.permute(0, 1);
        p2.permute(2, 3);
        p3.permute(0, 2).permute(1, 3);
        p4.permute(4, 5);
        p5.permute(6, 7);
        p6.permute(4, 6).permute(5, 7);
        p7.permute(0, 4).permute(1, 5).permute(2, 6).permute(3, 7);

        symmetry_element_set<8, double> set1(se_perm_t::k_sym_type);
        set1.insert(se_perm_t(p1, false));
        set1.insert(se_perm_t(p2, false));
        set1.insert(se_perm_t(p3, true));
        set1.insert(se_perm_t(p4, false));
        set1.insert(se_perm_t(p5, false));
        set1.insert(se_perm_t(p6, true));
        set1.insert(se_perm_t(p7, true));
        permutation_group<8, double> pg1(set1), pg2;

        mask<8> msk[2];
        msk[0][2] = true; msk[0][6] = true;
        msk[1][3] = true; msk[1][7] = true;
        pg1.stabilize(msk, pg2);

        if(!pg2.is_member(false, p1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg4.is_member(false, p1)");
        }
        if(!pg2.is_member(false, p4)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg4.is_member(false, p4)");
        }
        if(!pg2.is_member(true, p7)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!pg4.is_member(true, p7)");
        }

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
        const std::list< std::pair<permutation<N>, bool> > &lst)
throw(libtest::test_exception) {

    typedef std::pair<permutation<N>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;
    typedef typename perm_list_t::const_iterator const_iterator;

    for(const_iterator i = lst.begin(); i != lst.end(); i++) {
        for(const_iterator j = lst.begin(); j != lst.end(); j++) {

            permutation<N> p(i->first); p.permute(j->first);

            const_iterator k = lst.begin();
            for (; k != lst.end(); k++)
                if (k->first.equals(p)) break;

            if (k == lst.end()) {
                std::ostringstream ss;
                ss << "Not a group: missing " << p << "("
                        << (i->first == j->first ? '+' : '-') << ")";
                fail_test(testname, __FILE__, __LINE__,
                        ss.str().c_str());
            }

            if (k->second && (i->second != j->first)) {
                std::ostringstream ss;
                ss << "Not a group: " << p << "("
                        << (i->first == j->first ? '+' : '-') << ") is invalid.";
                fail_test(testname, __FILE__, __LINE__,
                        ss.str().c_str());
            }
        }
    }
}


template<size_t N, typename T>
void permutation_group_test::verify_members(const char *testname,
        const permutation_group<N, T> &grp,
        const std::list< std::pair<permutation<N>, bool> > &allowed)
throw(libtest::test_exception) {

    typedef std::pair<permutation<N>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;
    typedef typename perm_list_t::iterator iterator;
    typedef typename perm_list_t::const_iterator const_iterator;

    perm_list_t all;
    all_permutations(false, all);
    for(iterator i = all.begin(); i != all.end(); i++) {
        if (! i->first.is_identity() && ! i->second)
            all.push_back(signed_perm_t(i->first, true));
    }

    for(iterator i = all.begin(); i != all.end(); i++) {

        const_iterator isp = allowed.begin();
        for (; isp != allowed.end(); isp++)
            if (isp->first.equals(i->first) &&
                    isp->second == i->second) break;
        bool bsp = grp.is_member(i->second, i->first);
        if(bsp != (isp != allowed.end())) {
            std::ostringstream ss;
            ss << "Inconsistent: " << i->first << "("
                    << (i->second ? '+' : '-') << ") ";
            if(bsp) ss << "should not be allowed.";
            else ss << "should be allowed.";
            fail_test(testname, __FILE__, __LINE__,
                    ss.str().c_str());
        }
    }
}


template<size_t N, typename T>
void permutation_group_test::verify_genset(const char *testname,
        const permutation_group<N, T> &grp,
        const std::list< std::pair<permutation<N>, bool> > &allowed)
throw(libtest::test_exception) {

    typedef std::pair<permutation<N>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;
    typedef typename perm_list_t::iterator iterator;
    typedef typename perm_list_t::const_iterator const_iterator;

    symmetry_element_set<N, T> set(se_perm<N, T>::k_sym_type);
    grp.convert(set);
    symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter(set);

    perm_list_t lst;
    gen_group(adapter, true, permutation<N>(), lst);

    if(lst.size() != allowed.size()) {
        fail_test(testname, __FILE__, __LINE__,
                "Unexpected group size");
    }

    for(const_iterator i = lst.begin(); i != lst.end(); i++) {

        const_iterator is = lst.begin();
        for (; is != lst.end(); is++)
            if (is->first.equals(i->first) && is->second == i->second) break;
        if(is == allowed.end()) {

            std::ostringstream ss;
            ss << "Permutation " << i->first << "("
                    << (i->second ? '+' : '-') << " is not allowed.";
            fail_test(testname, __FILE__, __LINE__,
                    ss.str().c_str());
        }
    }

}


template<size_t N>
void permutation_group_test::all_permutations(
        bool sign, std::list< std::pair<permutation<N>, bool> > &lst) {

    typedef std::pair<permutation<N>, bool> signed_perm_t;
    typedef std::list< std::pair<permutation<N - 1>, bool> > perm_list_t;

    perm_list_t lst2;
    all_permutations(sign, lst2);

    for(typename perm_list_t::iterator i2 = lst2.begin();
            i2 != lst2.end(); i2++) {

        sequence<N - 1, size_t> seq2a(0), seq2b(0);
        for(size_t j = 0; j < N - 1; j++) seq2a[j] = seq2b[j] = j;
        i2->first.apply(seq2b);

        for(size_t i = 0; i < N; i++) {

            sequence<N, size_t> seq1a(0), seq1b(0);

            for(size_t j = 0, k = 0; j < N; j++) {
                seq1a[j] = j;
                if(j == i) seq1b[j] = N - 1;
                else { seq1b[j] = seq2b[k]; k++; }
            }

            permutation_builder<N> pb1(seq1b, seq1a);
            if (sign)
                lst.push_back(signed_perm_t(pb1.get_perm(), true));
            else
                lst.push_back(signed_perm_t(pb1.get_perm(),
                        ((N - i) % 2 ? i2->second : ! i2->second)));
        }
    }
}


void permutation_group_test::all_permutations(
        bool sign, std::list< std::pair<permutation<1>, bool> > &lst) {

    typedef std::pair<permutation<1>, bool> signed_perm_t;

    lst.push_back(signed_perm_t(permutation<1>(), true));
}


void permutation_group_test::all_permutations(
        bool sign, std::list< std::pair< permutation<0>, bool> > &lst) {

}


template<size_t N, typename T>
void permutation_group_test::gen_group(
        const symmetry_element_set_adapter< N, T, se_perm<N, T> > &set,
        bool sign, const permutation<N> &perm0,
        std::list< std::pair<permutation<N>, bool> > &lst) {

    typedef std::pair<permutation<N>, bool> signed_perm_t;
    typedef std::list<signed_perm_t> perm_list_t;
    typedef typename perm_list_t::const_iterator const_iterator;

    const_iterator p = lst.begin();
    for(; p != lst.end(); p++)
        if (p->first.equals(perm0) && p->second == sign) return;

    if(sign || ! perm0.is_identity()) lst.push_back(signed_perm_t(perm0, sign));

    typename symmetry_element_set_adapter< N, T, se_perm<N, T> >::iterator i =
            set.begin();
    for(; i != set.end(); i++) {
        const se_perm<N, T> &e = set.get_elem(i);
        permutation<N> perm1(perm0);
        perm1.permute(e.get_perm());
        gen_group(set, (e.is_symm() ? sign : ! sign), perm1, lst);
    }
}


} // namespace libtensor
