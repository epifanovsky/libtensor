#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_random.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_random_test.h"

namespace libtensor {


void diag_tod_random_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


/** \test Tests diag_tod_random on a tensor with empty allowed space. Expected
        an unchanged tensor.
 **/
void diag_tod_random_test::test_1() {

    static const char *testname = "diag_tod_random_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dims66(index_range<2>(i2a, i2b));

        diag_tensor_space<2> dts(dims66);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims66);

        diag_tod_random<2>().perform(dt);
        tod_conv_diag_tensor<2>(dt).perform(t);

        {
            dense_tensor_ctrl<2, double> ct(t);
            const double *p = ct.req_const_dataptr();
            for(abs_index<2> ai(dims66); !ai.is_last(); ai.inc()) {
                bool nonzero = p[ai.get_abs_index()] != 0.0;
                const index<2> &i = ai.get_index();
                if(nonzero) {
                    std::ostringstream ss;
                    ss << "Unexpected non-zero entry at " << i << ".";
                    fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
                }
            }
            ct.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests diag_tod_random on a 2-dim tensor with one diagonal.
 **/
void diag_tod_random_test::test_2() {

    static const char *testname = "diag_tod_random_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dims66(index_range<2>(i2a, i2b));

        mask<2> m11;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts1(1);
        dts1.set_diag_mask(0, m11);

        diag_tensor_space<2> dts(dims66);
        dts.add_subspace(dts1);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims66);

        diag_tod_random<2>().perform(dt);
        tod_conv_diag_tensor<2>(dt).perform(t);

        {
            dense_tensor_ctrl<2, double> ct(t);
            const double *p = ct.req_const_dataptr();
            for(abs_index<2> ai(dims66); !ai.is_last(); ai.inc()) {
                bool nonzero = p[ai.get_abs_index()] != 0.0;
                const index<2> &i = ai.get_index();
                bool diag = i[0] == i[1];
                if(diag != nonzero) {
                    std::ostringstream ss;
                    ss << "Unexpected entry at " << i << ".";
                    fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
                }

            }
            ct.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests diag_tod_random on a 2-dim tensor with two trivial diagonals.
 **/
void diag_tod_random_test::test_3() {

    static const char *testname = "diag_tod_random_test::test_3()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dims66(index_range<2>(i2a, i2b));

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        diag_tensor_subspace<2> dts1(2);
        dts1.set_diag_mask(0, m01);
        dts1.set_diag_mask(1, m10);

        diag_tensor_space<2> dts(dims66);
        dts.add_subspace(dts1);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims66);

        diag_tod_random<2>().perform(dt);
        tod_conv_diag_tensor<2>(dt).perform(t);

        {
            dense_tensor_ctrl<2, double> ct(t);
            const double *p = ct.req_const_dataptr();
            for(abs_index<2> ai(dims66); !ai.is_last(); ai.inc()) {
                bool nonzero = p[ai.get_abs_index()] != 0.0;
                const index<2> &i = ai.get_index();
                if(!nonzero) {
                    std::ostringstream ss;
                    ss << "Unexpected zero entry at " << i << ".";
                    fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
                }
            }
            ct.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests diag_tod_random on a 2-dim tensor with two trivial diagonals
        (subspace 1) and one full diagonal (subspace 2).
 **/
void diag_tod_random_test::test_4() {

    static const char *testname = "diag_tod_random_test::test_4()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dims66(index_range<2>(i2a, i2b));

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dts1(1), dts2(2);
        dts1.set_diag_mask(0, m11);
        dts2.set_diag_mask(0, m01);
        dts2.set_diag_mask(1, m10);

        diag_tensor_space<2> dts(dims66);
        dts.add_subspace(dts1);
        dts.add_subspace(dts2);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims66);

        diag_tod_random<2>().perform(dt);
        tod_conv_diag_tensor<2>(dt).perform(t);

        {
            dense_tensor_ctrl<2, double> ct(t);
            const double *p = ct.req_const_dataptr();
            for(abs_index<2> ai(dims66); !ai.is_last(); ai.inc()) {
                bool nonzero = p[ai.get_abs_index()] != 0.0;
                const index<2> &i = ai.get_index();
                if(!nonzero) {
                    std::ostringstream ss;
                    ss << "Unexpected zero entry at " << i << ".";
                    fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
                }
            }
            ct.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests diag_tod_random on a 4-dim tensor with two diagonals
        (subspace 1) and one full diagonal (subspace 2).
 **/
void diag_tod_random_test::test_5() {

    static const char *testname = "diag_tod_random_test::test_5()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
        dimensions<4> dims6666(index_range<4>(i4a, i4b));

        mask<4> m0101, m1010, m1111;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        diag_tensor_subspace<4> dts1(1), dts2(2);
        dts1.set_diag_mask(0, m1111);
        dts2.set_diag_mask(0, m0101);
        dts2.set_diag_mask(1, m1010);

        diag_tensor_space<4> dts(dims6666);
        dts.add_subspace(dts1);
        dts.add_subspace(dts2);

        diag_tensor<4, double, allocator_t> dt(dts);
        dense_tensor<4, double, allocator_t> t(dims6666);

        diag_tod_random<4>().perform(dt);
        tod_conv_diag_tensor<4>(dt).perform(t);

        {
            dense_tensor_ctrl<4, double> ct(t);
            const double *p = ct.req_const_dataptr();
            for(abs_index<4> ai(dims6666); !ai.is_last(); ai.inc()) {
                bool nonzero = p[ai.get_abs_index()] != 0.0;
                bool diag = false;
                const index<4> &i = ai.get_index();
                if(i[0] == i[2] && i[1] == i[3]) diag = true;
                if(diag != nonzero) {
                    std::ostringstream ss;
                    ss << "Unexpected entry at " << i << ".";
                    fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
                }
            }
            ct.ret_const_dataptr(p);
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

