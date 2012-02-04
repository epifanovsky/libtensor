#include <libtensor/diag_tensor/diag_tensor_space.h>
#include "diag_tensor_space_test.h"

namespace libtensor {


void diag_tensor_space_test::perform() throw(libtest::test_exception) {

    test_1();

    test_exc_1();
}


void diag_tensor_space_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_space_test::test_1()";

    try {

        index<4> i1, i2;
        i2[0] = 5; i2[1] = 6; i2[2] = 5; i2[3] = 6;
        dimensions<4> dims(index_range<4>(i1, i2));

        mask<4> m0101, m1010;
        m0101[1] = true; m0101[3] = true;
        m1010[0] = true; m1010[2] = true;

        diag_tensor_subspace<4> dts1(1), dts2(1);
        dts1.set_diag_mask(0, m0101);
        dts2.set_diag_mask(0, m1010);

        diag_tensor_space<4> dts(dims);

        if(dts.get_nsubspaces() != 0) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_nsubspaces() != 0");
        }

        size_t ssn1 = dts.add_subspace(dts1);
        size_t ssn2 = dts.add_subspace(dts2);
        if(ssn2 == ssn1) {
            fail_test(testname, __FILE__, __LINE__, "ssn2 == ssn1");
        }

        if(dts.get_nsubspaces() != 2) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_nsubspaces() != 2");
        }
        if(dts.get_subspace_size(ssn1) != 7 * 36) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_subspace_size(ssn1) != 7 * 36");
        }
        if(dts.get_subspace_size(ssn2) != 6 * 49) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_subspace_size(ssn2) != 6 * 49");
        }

        std::vector<size_t> ssn;
        dts.get_all_subspaces(ssn);
        if(ssn.size() != 2) {
            fail_test(testname, __FILE__, __LINE__, "ssn.size() != 2");
        }
        std::vector<size_t>::iterator ssi1 =
            std::find(ssn.begin(), ssn.end(), ssn1);
        std::vector<size_t>::iterator ssi2 =
            std::find(ssn.begin(), ssn.end(), ssn2);
        if(ssi1 == ssn.end()) {
            fail_test(testname, __FILE__, __LINE__, "ssi1 == ssn.end()");
        }
        if(ssi2 == ssn.end()) {
            fail_test(testname, __FILE__, __LINE__, "ssi2 == ssn.end()");
        }
        
        if(!m0101.equals(dts.get_subspace(ssn1).get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "subspace 0101 not found");
        }
        dts.remove_subspace(ssn1);

        if(dts.get_nsubspaces() != 1) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_nsubspaces() != 1");
        }

        if(!m1010.equals(dts.get_subspace(ssn2).get_total_mask())) {
            fail_test(testname, __FILE__, __LINE__, "subspace 1010 not found");
        }
        dts.remove_subspace(ssn2);

        if(dts.get_nsubspaces() != 0) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_nsubspaces() != 0");
        }

        ssn1 = dts.add_subspace(dts1);
        if(dts.get_nsubspaces() != 1) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_nsubspaces() != 1");
        }

        ssn2 = dts.add_subspace(dts2);
        if(ssn2 == ssn1) {
            fail_test(testname, __FILE__, __LINE__, "ssn2 == ssn1");
        }

        if(dts.get_nsubspaces() != 2) {
            fail_test(testname, __FILE__, __LINE__,
                "dts.get_nsubspaces() != 2");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tensor_space_test::test_exc_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_space_test::test_exc_1()";

    bool ok = false;

    try {

        index<4> i1, i2;
        i2[0] = 5; i2[1] = 6; i2[2] = 5; i2[3] = 6;
        dimensions<4> dims(index_range<4>(i1, i2));

        mask<4> m0011;
        m0011[2] = true; m0011[3] = true;

        diag_tensor_subspace<4> dts1(1);
        dts1.set_diag_mask(0, m0011);

        diag_tensor_space<4> dts(dims);

        try {
            dts.add_subspace(dts1);
        } catch(bad_parameter&) {
            ok = true;
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected bad_parameter: "
            "diagonal connects incompatible indexes.");
    }
}


} // namespace libtensor

