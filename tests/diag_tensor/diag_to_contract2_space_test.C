#include <iostream>
#include <libtensor/diag_tensor/diag_to_contract2_space.h>
#include "diag_to_contract2_space_test.h"

namespace libtensor {


void diag_to_contract2_space_test::perform() throw(libtest::test_exception) {

    test_1a();
    test_1b();
    test_1c();
    test_2a();
    test_3a();
    test_4a();
    test_4b();
    test_4c();
}


void diag_to_contract2_space_test::test_1a() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_1a()";

    try {

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 5;
        dimensions<2> dims(index_range<2>(i1, i2));

        mask<2> m11;
        m11[0] = true; m11[1] = true;
        mask<4> m0011, m1100;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;

        diag_tensor_subspace<2> dtss1(1);
        dtss1.set_diag_mask(0, m11);
        diag_tensor_subspace<4> dtss3(2);
        dtss3.set_diag_mask(0, m0011);
        dtss3.set_diag_mask(1, m1100);

        diag_tensor_space<2> dtsa(dims), dtsb(dims);
        dtsa.add_subspace(dtss1);
        dtsb.add_subspace(dtss1);

        contraction2<2, 2, 0> contr;
        diag_to_contract2_space<2, 2, 0> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<4> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss3)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_1b() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_1b()";

    try {

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 5;
        dimensions<2> dims(index_range<2>(i1, i2));

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        mask<4> m1001;
        m1001[0] = true; m1001[3] = true;

        diag_tensor_subspace<2> dtss1(1), dtss2(2);
        dtss1.set_diag_mask(0, m11);
        dtss2.set_diag_mask(0, m01);
        dtss2.set_diag_mask(1, m10);
        diag_tensor_subspace<4> dtss3(1);
        dtss3.set_diag_mask(0, m1001);

        diag_tensor_space<2> dtsa(dims), dtsb(dims);
        dtsa.add_subspace(dtss1);
        dtsb.add_subspace(dtss2);

        permutation<4> permc;
        permc.permute(1, 2).permute(2, 3);
        contraction2<2, 2, 0> contr(permc);
        diag_to_contract2_space<2, 2, 0> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<4> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss3)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_1c() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_1c()";

    try {

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 5;
        dimensions<2> dims(index_range<2>(i1, i2));

        mask<2> m11;
        m11[0] = true; m11[1] = true;
        mask<4> m0011, m1100;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;

        diag_tensor_subspace<2> dtss1(1);
        dtss1.set_diag_mask(0, m11);
        diag_tensor_subspace<4> dtss2(2);
        dtss2.set_diag_mask(0, m0011);
        dtss2.set_diag_mask(1, m1100);

        diag_tensor_space<2> dtsa(dims), dtsb(dims);
        dtsa.add_subspace(dtss1);
        dtsb.add_subspace(dtss1);

        permutation<4> permc;
        permc.permute(0, 2).permute(1, 3);
        contraction2<2, 2, 0> contr(permc);
        diag_to_contract2_space<2, 2, 0> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<4> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss2)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_2a() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_2a()";

    try {

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 5;
        dimensions<2> dims(index_range<2>(i1, i2));

        mask<2> m11;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtss1(1);
        dtss1.set_diag_mask(0, m11);

        diag_tensor_space<2> dtsa(dims), dtsb(dims);
        dtsa.add_subspace(dtss1);
        dtsb.add_subspace(dtss1);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        diag_to_contract2_space<1, 1, 1> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<2> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss1)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_3a() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_3a()";

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dims2(index_range<2>(i2a, i2b));
        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
        dimensions<4> dims4(index_range<4>(i4a, i4b));

        mask<2> m11;
        m11[0] = true; m11[1] = true;
        mask<4> m0011, m1100;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;

        diag_tensor_subspace<2> dtss1(1);
        dtss1.set_diag_mask(0, m11);
        diag_tensor_subspace<4> dtss3(2);
        dtss3.set_diag_mask(0, m0011);
        dtss3.set_diag_mask(1, m1100);

        diag_tensor_space<4> dtsa(dims4);
        diag_tensor_space<2> dtsb(dims2);
        dtsa.add_subspace(dtss3);
        dtsb.add_subspace(dtss1);

        contraction2<2, 0, 2> contr;
        contr.contract(1, 0);
        contr.contract(3, 1);
        diag_to_contract2_space<2, 0, 2> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<2> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss1)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_4a() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_4a()";

    try {

        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
        dimensions<4> dims4(index_range<4>(i4a, i4b));

        mask<4> m0011, m1100, m0101, m1010, m0110, m1001;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1001[0] = true; m0110[1] = true; m0110[2] = true; m1001[3] = true;

        diag_tensor_subspace<4> dtss1(2), dtss2(2), dtss3(2);
        dtss1.set_diag_mask(0, m1100);
        dtss1.set_diag_mask(1, m0011);
        dtss2.set_diag_mask(0, m0101);
        dtss2.set_diag_mask(1, m1010);
        dtss3.set_diag_mask(0, m1001);
        dtss3.set_diag_mask(1, m0110);

        diag_tensor_space<4> dtsa(dims4), dtsb(dims4);
        dtsa.add_subspace(dtss1);
        dtsb.add_subspace(dtss2);

        permutation<4> permc;
        permc.permute(1, 2).permute(2, 3);
        contraction2<2, 2, 2> contr(permc);
        contr.contract(2, 3);
        contr.contract(3, 1);
        diag_to_contract2_space<2, 2, 2> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<4> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss3)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_4b() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_4b()";

    try {

        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
        dimensions<4> dims4(index_range<4>(i4a, i4b));

        mask<4> m0011, m1100, m0101, m1010, m0110, m1001;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1001[0] = true; m0110[1] = true; m0110[2] = true; m1001[3] = true;

        diag_tensor_subspace<4> dtss1(2), dtss2(2), dtss3(2);
        dtss1.set_diag_mask(0, m1100);
        dtss1.set_diag_mask(1, m0011);
        dtss2.set_diag_mask(0, m0101);
        dtss2.set_diag_mask(1, m1010);
        dtss3.set_diag_mask(0, m1001);
        dtss3.set_diag_mask(1, m0110);

        diag_tensor_space<4> dtsa(dims4), dtsb(dims4);
        dtsa.add_subspace(dtss3);
        dtsb.add_subspace(dtss2);

        permutation<4> permc;
        permc.permute(1, 2).permute(2, 3);
        contraction2<2, 2, 2> contr(permc);
        contr.contract(2, 3);
        contr.contract(3, 1);
        diag_to_contract2_space<2, 2, 2> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<4> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss3)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_to_contract2_space_test::test_4c() throw(libtest::test_exception) {

    static const char *testname = "diag_to_contract2_space_test::test_4c()";

    try {

        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
        dimensions<4> dims4(index_range<4>(i4a, i4b));

        mask<4> m0011, m1100, m0101, m1010, m0110, m1001, m1110, m1111;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1001[0] = true; m0110[1] = true; m0110[2] = true; m1001[3] = true;
        m1110[0] = true; m1110[1] = true; m1110[2] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        diag_tensor_subspace<4> dtss1(1), dtss2(2), dtss3(2), dtss4(1);
        dtss1.set_diag_mask(0, m1110);
        dtss2.set_diag_mask(0, m0101);
        dtss2.set_diag_mask(1, m1010);
        dtss3.set_diag_mask(0, m1001);
        dtss3.set_diag_mask(1, m0110);
        dtss4.set_diag_mask(0, m1111);

        diag_tensor_space<4> dtsa(dims4), dtsb(dims4);
        dtsa.add_subspace(dtss1);
        dtsb.add_subspace(dtss1);

        contraction2<2, 2, 2> contr;
        contr.contract(2, 1);
        contr.contract(3, 3);
        diag_to_contract2_space<2, 2, 2> dtcontr(contr, dtsa, dtsb);
        diag_tensor_space<4> dtsc(dtcontr.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 1");
        }
        if(!dtsc.get_subspace(ssl[0]).equals(dtss4)) {
            fail_test(testname, __FILE__, __LINE__, "bad dtsc");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

