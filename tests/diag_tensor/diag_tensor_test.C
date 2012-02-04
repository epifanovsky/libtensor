#include <libtensor/core/allocator.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include "diag_tensor_test.h"

namespace libtensor {


void diag_tensor_test::perform() throw(libtest::test_exception) {

    test_1();
}


void diag_tensor_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tensor_test::test_1()";

    typedef std_allocator<double> allocator_t;

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
        size_t ssn1 = dts.add_subspace(dts1);
        size_t ssn2 = dts.add_subspace(dts2);
        size_t sz1 = dts.get_subspace_size(ssn1);
        size_t sz2 = dts.get_subspace_size(ssn2);

        diag_tensor<4, double, allocator_t> dt(dts);

        {
            diag_tensor_wr_ctrl<4, double> ctrl(dt);
            double *p1 = ctrl.req_dataptr(ssn1);
            double *p2 = ctrl.req_dataptr(ssn2);
            for(size_t i = 0; i < sz1; i++) p1[i] = 0.1;
            for(size_t i = 0; i < sz2; i++) p2[i] = -0.1;
            ctrl.ret_dataptr(ssn1, p1); p1 = 0;
            ctrl.ret_dataptr(ssn2, p2); p2 = 0;
        }
        {
            diag_tensor_rd_ctrl<4, double> ctrl(dt);
            const double *p1 = ctrl.req_const_dataptr(ssn1);
            const double *p2 = ctrl.req_const_dataptr(ssn2);
            for(size_t i = 0; i < sz1; i++) if(p1[i] != 0.1) {
                fail_test(testname, __FILE__, __LINE__, "Bad data in p1.");
            }
            for(size_t i = 0; i < sz2; i++) if(p2[i] != -0.1) {
                fail_test(testname, __FILE__, __LINE__, "Bad data in p2.");
            }
            ctrl.ret_const_dataptr(ssn1, p1); p1 = 0;
            ctrl.ret_const_dataptr(ssn2, p2); p2 = 0;
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

