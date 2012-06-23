#include <libtensor/diag_tensor/diag_to_add_space.h>
#include "diag_to_add_space_test.h"

namespace libtensor {


void diag_to_add_space_test::perform() throw(libtest::test_exception) {

    test_1();
}


void diag_to_add_space_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "diag_to_add_space_test::test_1()";

    try {

        index<3> i1, i2;
        i2[0] = 5; i2[1] = 5; i2[2] = 5;
        dimensions<3> dims(index_range<3>(i1, i2));

        mask<3> m011, m110, m111;
        m011[1] = true; m011[2] = true;
        m110[0] = true; m110[1] = true;
        m111[0] = true; m111[1] = true; m111[2] = true;

        diag_tensor_subspace<3> dtss1(1), dtss2(1), dtss3(1);
        dtss1.set_diag_mask(0, m111);
        dtss2.set_diag_mask(0, m011);
        dtss3.set_diag_mask(0, m110);

        diag_tensor_space<3> dtsa(dims), dtsb(dims);
        dtsa.add_subspace(dtss1);
        dtsa.add_subspace(dtss2);
        dtsb.add_subspace(dtss1);
        dtsb.add_subspace(dtss3);

        diag_to_add_space<3> dtadd(dtsa, dtsb);
        diag_tensor_space<3> dtsc(dtadd.get_dtsc());

        std::vector<size_t> ssl;
        dtsc.get_all_subspaces(ssl);
        if(ssl.size() != 3) {
            fail_test(testname, __FILE__, __LINE__, "ssl.size() != 3");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

