#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_copy_wnd.h>
#include "../compare_ref.h"
#include "../test_utils.h"

using namespace libtensor;


int test_1() {

    static const char testname[] = "tod_copy_wnd_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<2> ia1, ia2;
        ia2[0] = 5; ia2[1] = 7;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        libtensor::index<2> ira1, ira2;
        ira2[0] = 5; ira2[1] = 7;
        index_range<2> ira(ira1, ira2);
        libtensor::index<2> ib1, ib2;
        ib2[0] = 5; ib2[1] = 7;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        libtensor::index<2> irb1, irb2;
        irb2[0] = 5; irb2[1] = 7;
        index_range<2> irb(irb1, irb2);

        dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            size_t sza = dimsa.get_size();
            size_t szb = dimsb.get_size();

            linalg::rng_set_i_x(0, sza, dta, 1, 1.0);
            linalg::rng_set_i_x(0, szb, dtb1, 1, 1.0);
            for(size_t i = 0; i < szb; i++) dtb2[i] = dtb1[i];

            for(size_t ia = ira1[0]; ia <= ira2[0]; ia++)
            for(size_t ja = ira1[1]; ja <= ira2[1]; ja++) {

                libtensor::index<2> idxa, idxb;
                idxa[0] = ia;
                idxa[1] = ja;
                idxb[0] = ia + irb1[0] - ira1[0];
                idxb[1] = ja + irb1[1] - ira1[1];
                size_t ija = abs_index<2>::get_abs_index(idxa, dimsa);
                size_t ijb = abs_index<2>::get_abs_index(idxb, dimsb);
                dtb2[ijb] = dta[ija];
            }

            tca.ret_dataptr(dta); dta = 0;
            tcb.ret_dataptr(dtb1); dtb1 = 0;
            tcb_ref.ret_dataptr(dtb2); dtb2 = 0;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        tod_copy_wnd<2>(ta, ira).perform(tb, irb);

        compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_2() {

    static const char testname[] = "tod_copy_wnd_test::test_2()";

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<2> ia1, ia2;
        ia2[0] = 7; ia2[1] = 9;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        libtensor::index<2> ira1, ira2;
        ira1[0] = 1; ira1[1] = 1;
        ira2[0] = 6; ira2[1] = 8;
        index_range<2> ira(ira1, ira2);
        libtensor::index<2> ib1, ib2;
        ib2[0] = 5; ib2[1] = 7;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        libtensor::index<2> irb1, irb2;
        irb2[0] = 5; irb2[1] = 7;
        index_range<2> irb(irb1, irb2);

        dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            size_t sza = dimsa.get_size();
            size_t szb = dimsb.get_size();

            linalg::rng_set_i_x(0, sza, dta, 1, 1.0);
            linalg::rng_set_i_x(0, szb, dtb1, 1, 1.0);
            for(size_t i = 0; i < szb; i++) dtb2[i] = dtb1[i];

            for(size_t ia = ira1[0]; ia <= ira2[0]; ia++)
            for(size_t ja = ira1[1]; ja <= ira2[1]; ja++) {

                libtensor::index<2> idxa, idxb;
                idxa[0] = ia;
                idxa[1] = ja;
                idxb[0] = ia + irb1[0] - ira1[0];
                idxb[1] = ja + irb1[1] - ira1[1];
                size_t ija = abs_index<2>::get_abs_index(idxa, dimsa);
                size_t ijb = abs_index<2>::get_abs_index(idxb, dimsb);
                dtb2[ijb] = dta[ija];
            }

            tca.ret_dataptr(dta); dta = 0;
            tcb.ret_dataptr(dtb1); dtb1 = 0;
            tcb_ref.ret_dataptr(dtb2); dtb2 = 0;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        tod_copy_wnd<2>(ta, ira).perform(tb, irb);

        compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_3() {

    static const char testname[] = "tod_copy_wnd_test::test_3()";

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<2> ia1, ia2;
        ia2[0] = 5; ia2[1] = 7;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        libtensor::index<2> ira1, ira2;
        ira2[0] = 5; ira2[1] = 7;
        index_range<2> ira(ira1, ira2);
        libtensor::index<2> ib1, ib2;
        ib2[0] = 7; ib2[1] = 9;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        libtensor::index<2> irb1, irb2;
        irb1[0] = 1; irb1[1] = 2;
        irb2[0] = 6; irb2[1] = 9;
        index_range<2> irb(irb1, irb2);

        dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            size_t sza = dimsa.get_size();
            size_t szb = dimsb.get_size();

            linalg::rng_set_i_x(0, sza, dta, 1, 1.0);
            linalg::rng_set_i_x(0, szb, dtb1, 1, 1.0);
            for(size_t i = 0; i < szb; i++) dtb2[i] = dtb1[i];

            for(size_t ia = ira1[0]; ia <= ira2[0]; ia++)
            for(size_t ja = ira1[1]; ja <= ira2[1]; ja++) {

                libtensor::index<2> idxa, idxb;
                idxa[0] = ia;
                idxa[1] = ja;
                idxb[0] = ia + irb1[0] - ira1[0];
                idxb[1] = ja + irb1[1] - ira1[1];
                size_t ija = abs_index<2>::get_abs_index(idxa, dimsa);
                size_t ijb = abs_index<2>::get_abs_index(idxb, dimsb);
                dtb2[ijb] = dta[ija];
            }

            tca.ret_dataptr(dta); dta = 0;
            tcb.ret_dataptr(dtb1); dtb1 = 0;
            tcb_ref.ret_dataptr(dtb2); dtb2 = 0;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        tod_copy_wnd<2>(ta, ira).perform(tb, irb);

        compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_4() {

    static const char testname[] = "tod_copy_wnd_test::test_4()";

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<2> ia1, ia2;
        ia2[0] = 5; ia2[1] = 7;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        libtensor::index<2> ira1, ira2;
        ira1[0] = 0; ira1[1] = 3;
        ira2[0] = 3; ira2[1] = 6;
        index_range<2> ira(ira1, ira2);
        libtensor::index<2> ib1, ib2;
        ib2[0] = 5; ib2[1] = 7;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        libtensor::index<2> irb1, irb2;
        irb1[0] = 2; irb1[1] = 1;
        irb2[0] = 5; irb2[1] = 4;
        index_range<2> irb(irb1, irb2);

        dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            size_t sza = dimsa.get_size();
            size_t szb = dimsb.get_size();

            linalg::rng_set_i_x(0, sza, dta, 1, 1.0);
            linalg::rng_set_i_x(0, szb, dtb1, 1, 1.0);
            for(size_t i = 0; i < szb; i++) dtb2[i] = dtb1[i];

            for(size_t ia = ira1[0]; ia <= ira2[0]; ia++)
            for(size_t ja = ira1[1]; ja <= ira2[1]; ja++) {

                libtensor::index<2> idxa, idxb;
                idxa[0] = ia;
                idxa[1] = ja;
                idxb[0] = ia + irb1[0] - ira1[0];
                idxb[1] = ja + irb1[1] - ira1[1];
                size_t ija = abs_index<2>::get_abs_index(idxa, dimsa);
                size_t ijb = abs_index<2>::get_abs_index(idxb, dimsb);
                dtb2[ijb] = dta[ija];
            }

            tca.ret_dataptr(dta); dta = 0;
            tcb.ret_dataptr(dtb1); dtb1 = 0;
            tcb_ref.ret_dataptr(dtb2); dtb2 = 0;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        tod_copy_wnd<2>(ta, ira).perform(tb, irb);

        compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_5() {

    static const char testname[] = "tod_copy_wnd_test::test_5()";

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<4> ia1, ia2;
        ia2[0] = 5; ia2[1] = 7; ia2[2] = 5; ia2[3] = 7;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        libtensor::index<4> ira1, ira2;
        ira2[0] = 5; ira2[1] = 7; ira2[2] = 5; ira2[3] = 7;
        index_range<4> ira(ira1, ira2);
        libtensor::index<4> ib1, ib2;
        ib2[0] = 5; ib2[1] = 7; ib2[2] = 5; ib2[3] = 7;
        dimensions<4> dimsb(index_range<4>(ib1, ib2));
        libtensor::index<4> irb1, irb2;
        irb2[0] = 5; irb2[1] = 7; irb2[2] = 5; irb2[3] = 7;
        index_range<4> irb(irb1, irb2);

        dense_tensor<4, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<4, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            size_t sza = dimsa.get_size();
            size_t szb = dimsb.get_size();

            linalg::rng_set_i_x(0, sza, dta, 1, 1.0);
            linalg::rng_set_i_x(0, szb, dtb1, 1, 1.0);
            for(size_t i = 0; i < szb; i++) dtb2[i] = dtb1[i];

            for(size_t ia = ira1[0]; ia <= ira2[0]; ia++)
            for(size_t ja = ira1[1]; ja <= ira2[1]; ja++)
            for(size_t ka = ira1[2]; ka <= ira2[2]; ka++)
            for(size_t la = ira1[3]; la <= ira2[3]; la++) {

                libtensor::index<4> idxa, idxb;
                idxa[0] = ia;
                idxa[1] = ja;
                idxa[2] = ka;
                idxa[3] = la;
                idxb[0] = ia + irb1[0] - ira1[0];
                idxb[1] = ja + irb1[1] - ira1[1];
                idxb[2] = ka + irb1[2] - ira1[2];
                idxb[3] = la + irb1[3] - ira1[3];
                size_t ija = abs_index<4>::get_abs_index(idxa, dimsa);
                size_t ijb = abs_index<4>::get_abs_index(idxb, dimsb);
                dtb2[ijb] = dta[ija];
            }

            tca.ret_dataptr(dta); dta = 0;
            tcb.ret_dataptr(dtb1); dtb1 = 0;
            tcb_ref.ret_dataptr(dtb2); dtb2 = 0;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        tod_copy_wnd<4>(ta, ira).perform(tb, irb);

        compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    linalg::rng_setup(0);

    return

    test_1() |
    test_2() |
    test_3() |
    test_4() |
    test_5() |

    0;
}


