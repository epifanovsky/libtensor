#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include "se_perm_test.h"

namespace libtensor {


void se_perm_test::perform() throw(libtest::test_exception) {

    test_sym_ab_ba();
    test_asym_ab_ba();
    test_sym_abc_bca();
    test_asym_abc_bca();
    test_sym_abcd_badc();
    test_asym_abcd_badc();
}


/**	\test Tests the ab->ba permutational symmetry element
 **/
void se_perm_test::test_sym_ab_ba() throw(libtest::test_exception) {

    static const char *testname = "se_perm_test::test_sym_ab_ba()";

    try {

        permutation<2> perm;
        perm.permute(0, 1);
        se_perm<2, double> elem(perm, true);

        if(!elem.is_symm()) {
            fail_test(testname, __FILE__, __LINE__, "!elem.is_symm()");
        }
        if(!elem.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.get_perm().equals(perm)");
        }

        const transf<2, double> &tr = elem.get_transf();

        if(!tr.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect transformation permutation.");
        }
        if(tr.get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect transformation coefficient.");
        }

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 5;
        dimensions<2> dims1(index_range<2>(i1, i2));
        block_index_space<2> bis1(dims1);

        if(!elem.is_valid_bis(bis1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis1)");
        }

        mask<2> m1; m1[0] = true; m1[1] = true;
        block_index_space<2> bis2(bis1);
        bis2.split(m1, 2);

        if(!elem.is_valid_bis(bis2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis2)");
        }

        i2[0] = 5; i2[1] = 6;
        dimensions<2> dims3(index_range<2>(i1, i2));
        block_index_space<2> bis3(dims3);

        if(elem.is_valid_bis(bis3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem.is_valid_bis(bis3)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the ab->ba permutational anti-symmetry element
 **/
void se_perm_test::test_asym_ab_ba() throw(libtest::test_exception) {

    static const char *testname = "se_perm_test::test_asym_ab_ba()";

    try {

        permutation<2> perm;
        perm.permute(0, 1);
        se_perm<2, double> elem(perm, false);

        if(elem.is_symm()) {
            fail_test(testname, __FILE__, __LINE__, "elem.is_symm()");
        }
        if(!elem.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.get_perm().equals(perm)");
        }

        const transf<2, double> &tr = elem.get_transf();

        if(!tr.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect transformation permutation.");
        }
        if(tr.get_coeff() != -1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect transformation coefficient.");
        }

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 5;
        dimensions<2> dims1(index_range<2>(i1, i2));
        block_index_space<2> bis1(dims1);

        if(!elem.is_valid_bis(bis1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis1)");
        }

        mask<2> m1; m1[0] = true; m1[1] = true;
        block_index_space<2> bis2(bis1);
        bis2.split(m1, 2);

        if(!elem.is_valid_bis(bis2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis2)");
        }

        i2[0] = 5; i2[1] = 6;
        dimensions<2> dims3(index_range<2>(i1, i2));
        block_index_space<2> bis3(dims3);

        if(elem.is_valid_bis(bis3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem.is_valid_bis(bis3)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the abc->bca permutational symmetry element
 **/
void se_perm_test::test_sym_abc_bca() throw(libtest::test_exception) {

    static const char *testname = "se_perm_test::test_sym_abc_bca()";

    try {

        permutation<3> perm;
        perm.permute(0, 1).permute(1, 2);
        se_perm<3, double> elem(perm, true);

        if(!elem.is_symm()) {
            fail_test(testname, __FILE__, __LINE__, "!elem.is_symm()");
        }
        if(!elem.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.get_perm().equals(perm)");
        }

        const transf<3, double> &tr = elem.get_transf();

        if(!tr.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr.get_perm().equals(perm)");
        }
        if(tr.get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr.get_coeff() != 1.0");
        }

        index<3> i1, i2;
        i2[0] = 5; i2[1] = 5; i2[2] = 5;
        dimensions<3> dims1(index_range<3>(i1, i2));
        block_index_space<3> bis1(dims1);

        if(!elem.is_valid_bis(bis1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis1)");
        }

        mask<3> m1; m1[0] = true; m1[1] = true; m1[2] = true;
        block_index_space<3> bis2(bis1);
        bis2.split(m1, 2);

        if(!elem.is_valid_bis(bis2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis2)");
        }

        i2[0] = 5; i2[1] = 6; i2[2] = 5;
        dimensions<3> dims3(index_range<3>(i1, i2));
        block_index_space<3> bis3(dims3);

        if(elem.is_valid_bis(bis3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem.is_valid_bis(bis3)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the abc->bca permutational anti-symmetry element.
 **/
void se_perm_test::test_asym_abc_bca() throw(libtest::test_exception) {

    static const char *testname = "se_perm_test::test_asym_abc_bca()";

    try {

        permutation<3> perm;
        perm.permute(0, 1).permute(1, 2);
        se_perm<3, double> elem(perm, false);

        if(elem.is_symm()) {
            fail_test(testname, __FILE__, __LINE__, "elem.is_symm()");
        }
        if(!elem.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.get_perm().equals(perm)");
        }

        const transf<3, double> &tr = elem.get_transf();

        if(!tr.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr.get_perm().equals(perm)");
        }
        if(tr.get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr.get_coeff() != 1.0");
        }

        index<3> i1, i2;
        i2[0] = 5; i2[1] = 5; i2[2] = 5;
        dimensions<3> dims1(index_range<3>(i1, i2));
        block_index_space<3> bis1(dims1);

        if(!elem.is_valid_bis(bis1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis1)");
        }

        mask<3> m1; m1[0] = true; m1[1] = true; m1[2] = true;
        block_index_space<3> bis2(bis1);
        bis2.split(m1, 2);

        if(!elem.is_valid_bis(bis2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis2)");
        }

        i2[0] = 5; i2[1] = 6; i2[2] = 5;
        dimensions<3> dims3(index_range<3>(i1, i2));
        block_index_space<3> bis3(dims3);

        if(elem.is_valid_bis(bis3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem.is_valid_bis(bis3)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the abcd->badc permutational symmetry element
 **/
void se_perm_test::test_sym_abcd_badc() throw(libtest::test_exception) {

    static const char *testname = "se_perm_test::test_sym_abcd_badc()";

    try {

        permutation<4> perm;
        perm.permute(0, 1).permute(2, 3);
        se_perm<4, double> elem(perm, true);

        if(!elem.is_symm()) {
            fail_test(testname, __FILE__, __LINE__, "!elem.is_symm()");
        }
        if(!elem.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.get_perm().equals(perm)");
        }

        const transf<4, double> &tr = elem.get_transf();

        if(!tr.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr.get_perm().equals(perm)");
        }
        if(tr.get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr.get_coeff() != 1.0");
        }

        index<4> i1, i2;
        i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 5;
        dimensions<4> dims1(index_range<4>(i1, i2));
        block_index_space<4> bis1(dims1);

        if(!elem.is_valid_bis(bis1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis1)");
        }

        mask<4> m1; m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
        block_index_space<4> bis2(bis1);
        bis2.split(m1, 2);

        if(!elem.is_valid_bis(bis2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis2)");
        }

        i2[0] = 5; i2[1] = 5; i2[2] = 6; i2[3] = 6;
        dimensions<4> dims3(index_range<4>(i1, i2));
        block_index_space<4> bis3(dims3);

        if(!elem.is_valid_bis(bis3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis3)");
        }

        i2[0] = 5; i2[1] = 6; i2[2] = 5; i2[3] = 6;
        dimensions<4> dims4(index_range<4>(i1, i2));
        block_index_space<4> bis4(dims4);

        if(elem.is_valid_bis(bis4)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem.is_valid_bis(bis4)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Tests the abcd->badc permutational anti-symmetry element
 **/
void se_perm_test::test_asym_abcd_badc() throw(libtest::test_exception) {

    static const char *testname = "se_perm_test::test_asym_abcd_badc()";

    try {

        permutation<4> perm;
        perm.permute(0, 1).permute(2, 3);
        se_perm<4, double> elem(perm, false);

        if(elem.is_symm()) {
            fail_test(testname, __FILE__, __LINE__, "elem.is_symm()");
        }
        if(!elem.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.get_perm().equals(perm)");
        }

        const transf<4, double> &tr = elem.get_transf();

        if(!tr.get_perm().equals(perm)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr.get_perm().equals(perm)");
        }
        if(tr.get_coeff() != -1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr.get_coeff() != -1.0");
        }

        index<4> i1, i2;
        i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 5;
        dimensions<4> dims1(index_range<4>(i1, i2));
        block_index_space<4> bis1(dims1);

        if(!elem.is_valid_bis(bis1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis1)");
        }

        mask<4> m1; m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
        block_index_space<4> bis2(bis1);
        bis2.split(m1, 2);

        if(!elem.is_valid_bis(bis2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis2)");
        }

        i2[0] = 5; i2[1] = 5; i2[2] = 6; i2[3] = 6;
        dimensions<4> dims3(index_range<4>(i1, i2));
        block_index_space<4> bis3(dims3);

        if(!elem.is_valid_bis(bis3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem.is_valid_bis(bis3)");
        }

        i2[0] = 5; i2[1] = 6; i2[2] = 5; i2[3] = 6;
        dimensions<4> dims4(index_range<4>(i1, i2));
        block_index_space<4> bis4(dims4);

        if(elem.is_valid_bis(bis4)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem.is_valid_bis(bis4)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

