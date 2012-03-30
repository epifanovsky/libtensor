#include <libtensor/diag_tensor/diag_tensor_space.h>
#include "diag_tensor_subspace_test.h"

namespace libtensor {


void diag_tensor_subspace_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();

    test_equals_1();
    test_equals_2();
    test_equals_3();

    test_permute_1();
    test_permute_2();
    test_permute_3();

    test_exc_1();
    test_exc_2();
    test_exc_3();
}


void diag_tensor_subspace_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_1()";

    try {

        mask<1> m0, m1;
        m1[0] = true;

        diag_tensor_subspace<1> dts1(0);

        if(dts1.get_ndiag() != 0) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_ndiag() != 0");
        }
        if(!m0.equals(dts1.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_total_mask()");
        }

        diag_tensor_subspace<1> dts2(1);

        if(dts2.get_ndiag() != 1) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_ndiag() != 0");
        }
        if(!m0.equals(dts2.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_total_mask()");
        }
        if(!m0.equals(dts2.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_diag_mask(0)");
        }
        dts2.set_diag_mask(0, m1);
        if(!m1.equals(dts2.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_total_mask()");
        }
        if(!m1.equals(dts2.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_diag_mask(0)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_2()";

    try {

        mask<2> m00, m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts1(0);

        if(dts1.get_ndiag() != 0) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_ndiag() != 0");
        }
        if(!m00.equals(dts1.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_total_mask()");
        }

        diag_tensor_subspace<2> dts2(1);

        if(dts2.get_ndiag() != 1) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_ndiag() != 1");
        }
        if(!m00.equals(dts2.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_total_mask()");
        }
        dts2.set_diag_mask(0, m11);
        if(!m11.equals(dts2.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_total_mask()");
        }
        if(!m11.equals(dts2.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts2.get_diag_mask(0)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_3()";

    try {

        mask<6> m000000, m010001, m010101, m101000, m111001, m111101;
        m010001[1] = true; m010001[5] = true;
        m010101[1] = true; m010101[3] = true; m010101[5] = true;
        m101000[0] = true; m101000[2] = true;
        m111001[0] = true; m111001[1] = true; m111001[2] = true;
        m111001[5] = true;
        m111101[0] = true; m111101[1] = true; m111101[2] = true;
        m111101[3] = true; m111101[5] = true;

        diag_tensor_subspace<6> dts1(2);

        if(dts1.get_ndiag() != 2) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_ndiag() != 2");
        }
        if(!m000000.equals(dts1.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_total_mask()");
        }
        if(!m000000.equals(dts1.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(0)");
        }
        if(!m000000.equals(dts1.get_diag_mask(1))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(1)");
        }

        dts1.set_diag_mask(0, m010101);

        if(dts1.get_ndiag() != 2) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_ndiag() != 2");
        }
        if(!m010101.equals(dts1.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_total_mask()");
        }
        if(!m010101.equals(dts1.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(0)");
        }
        if(!m000000.equals(dts1.get_diag_mask(1))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(1)");
        }

        dts1.set_diag_mask(1, m101000);

        if(dts1.get_ndiag() != 2) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_ndiag() != 2");
        }
        if(!m111101.equals(dts1.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_total_mask()");
        }
        if(!m010101.equals(dts1.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(0)");
        }
        if(!m101000.equals(dts1.get_diag_mask(1))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(1)");
        }

        dts1.set_diag_mask(0, m010001);

        if(dts1.get_ndiag() != 2) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_ndiag() != 2");
        }
        if(!m111001.equals(dts1.get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_total_mask()");
        }
        if(!m010001.equals(dts1.get_diag_mask(0))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(0)");
        }
        if(!m101000.equals(dts1.get_diag_mask(1))) {
            fail_test(testname, __FILE__, __LINE__, "dts1.get_diag_mask(1)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_equals_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_equals_1()";

    try {

        mask<1> m0, m1;
        m1[0] = true;

        diag_tensor_subspace<1> dts1(1), dts2(1);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }

        dts1.set_diag_mask(0, m1);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }

        dts2.set_diag_mask(0, m1);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_equals_2() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_equals_2()";

    try {

        mask<2> m00, m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts1(2), dts2(2), dts3(1);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts1.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts3)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }
        if(!dts3.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts3.equals(dts2)");
        }

        dts3.set_diag_mask(0, m11);

        if(dts1.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "dts1.equals(dts3)");
        }
        if(dts3.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "dts3.equals(dts2)");
        }
        if(!dts3.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "!dts3.equals(dts3)");
        }

        dts1.set_diag_mask(0, m10);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }
        if(dts1.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "dts1.equals(dts3)");
        }

        dts2.set_diag_mask(0, m01);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }
        if(dts2.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "dts2.equals(dts3)");
        }

        dts1.set_diag_mask(0, m01);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }
        if(dts3.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "dts3.equals(dts1)");
        }

        dts2.set_diag_mask(0, m10);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }
        if(dts3.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "dts3.equals(dts2)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_equals_3() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_equals_3()";

    try {

        mask<4> m0000, m0011, m1000, m0100;
        m0011[2] = true; m0011[3] = true;
        m1000[0] = true; m0100[1] = true;

        diag_tensor_subspace<4> dts1(1), dts2(3);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }

        dts1.set_diag_mask(0, m0011);
        dts2.set_diag_mask(1, m0011);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }

        dts2.set_diag_mask(0, m1000);
        dts2.set_diag_mask(2, m0100);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts2.equals(dts1)) {
            fail_test(testname, __FILE__, __LINE__, "!dts2.equals(dts1)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_permute_1()
    throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_permute_1()";

    try {

        mask<2> m00, m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts1(2), dts2(2), dts3(2), dts4(0), dts5(0);

        permutation<2> p01, p10;
        p10.permute(0, 1);

        dts1.set_diag_mask(0, m10);
        dts1.set_diag_mask(1, m01);
        dts2.set_diag_mask(0, m10);
        dts2.set_diag_mask(1, m01);
        dts3.set_diag_mask(0, m10);
        dts3.set_diag_mask(1, m01);

        dts2.permute(p01);
        dts3.permute(p10);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts1.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts3)");
        }

        dts4.permute(p01);
        dts5.permute(p10);

        if(!dts1.equals(dts4)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts4)");
        }
        if(!dts1.equals(dts5)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts5)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_permute_2()
    throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_permute_2()";

    try {

        mask<2> m00, m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts1(1), dts2(1), dts3(1);

        permutation<2> p01, p10;
        p10.permute(0, 1);

        dts1.set_diag_mask(0, m11);
        dts2.set_diag_mask(0, m11);
        dts3.set_diag_mask(0, m11);

        dts2.permute(p01);
        dts3.permute(p10);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts1.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts3)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_permute_3()
    throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_permute_3()";

    try {

        mask<3> m001, m010, m101, m110;
        m101[0] = true; m010[1] = true; m101[2] = true;
        m110[0] = true; m110[1] = true; m001[2] = true;

        diag_tensor_subspace<3> dts1(2), dts2(2), dts3(1), dts4(1);

        permutation<3> p012, p120, p021;
        p120.permute(0, 1).permute(1, 2);
        p021.permute(1, 2);

        dts1.set_diag_mask(0, m101);
        dts1.set_diag_mask(1, m010);
        dts2.set_diag_mask(0, m110);
        dts2.set_diag_mask(1, m001);
        dts3.set_diag_mask(0, m110);
        dts4.set_diag_mask(0, m110);

        dts1.permute(p012);
        dts2.permute(p120);
        dts3.permute(p120);
        dts4.permute(p021);

        if(!dts1.equals(dts2)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts2)");
        }
        if(!dts1.equals(dts3)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts3)");
        }
        if(!dts1.equals(dts4)) {
            fail_test(testname, __FILE__, __LINE__, "!dts1.equals(dts4)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_subspace_test::test_exc_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_exc_1()";

#ifdef LIBTENSOR_DEBUG

    bool ok = false;

    try {

        diag_tensor_subspace<1> dts(2);

    } catch(bad_parameter &e) {
        ok = true;
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected bad_parameter: "
            "number of diagonals exceeds tensor order.");
    }

#endif // LIBTENSOR_DEBUG
}


void diag_tensor_subspace_test::test_exc_2() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_exc_2()";

#ifdef LIBTENSOR_DEBUG

    bool ok = false;

    try {

        mask<3> m110, m011;
        m110[0] = true; m110[1] = true;
        m011[1] = true; m011[2] = true;

        diag_tensor_subspace<3> dts(2);
        dts.set_diag_mask(0, m110);
        try {
            dts.set_diag_mask(1, m011);
        } catch(bad_parameter&) {
            ok = true;
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected bad_parameter: "
            "overlapping diagonal masks.");
    }

#endif // LIBTENSOR_DEBUG
}


void diag_tensor_subspace_test::test_exc_3() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_subspace_test::test_exc_3()";

#ifdef LIBTENSOR_DEBUG

    bool ok = false;

    try {

        mask<2> m11;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts(1);
        try {
            dts.set_diag_mask(1, m11);
        } catch(out_of_bounds&) {
            ok = true;
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected out_of_bounds: "
            "diagonal index is too large.");
    }

#endif // LIBTENSOR_DEBUG
}


} // namespace libtensor

