#include <sstream>
#include <vector>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_mult1.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_mult1.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_mult1_test.h"

namespace libtensor {


void diag_tod_mult1_test::perform() throw(libtest::test_exception) {

    test_ij_1(false, true, 1);
    test_ij_1(false, true, 4);
    test_ij_1(false, true, 17);
    test_ij_1(true, true, 1);
    test_ij_1(true, true, 4);
    test_ij_1(true, true, 17);
    test_ij_1(false, false, 1);
    test_ij_1(false, false, 4);
    test_ij_1(false, false, 17);
    test_ij_1(true, false, 1);
    test_ij_1(true, false, 4);
    test_ij_1(true, false, 17);

    test_ij_2(false, true, 1);
    test_ij_2(false, true, 4);
    test_ij_2(false, true, 17);
    test_ij_2(true, true, 1);
    test_ij_2(true, true, 4);
    test_ij_2(true, true, 17);
    test_ij_2(false, false, 1);
    test_ij_2(false, false, 4);
    test_ij_2(false, false, 17);
    test_ij_2(true, false, 1);
    test_ij_2(true, false, 4);
    test_ij_2(true, false, 17);

    test_ij_3(false, true, 1);
    test_ij_3(false, true, 4);
    test_ij_3(false, true, 17);
    test_ij_3(true, true, 1);
    test_ij_3(true, true, 4);
    test_ij_3(true, true, 17);
    test_ij_3(false, false, 1);
    test_ij_3(false, false, 4);
    test_ij_3(false, false, 17);
    test_ij_3(true, false, 1);
    test_ij_3(true, false, 4);
    test_ij_3(true, false, 17);

    test_ij_4(false, true, 1);
    test_ij_4(false, true, 4);
    test_ij_4(false, true, 17);
    test_ij_4(true, true, 1);
    test_ij_4(true, true, 4);
    test_ij_4(true, true, 17);
    test_ij_4(false, false, 1);
    test_ij_4(false, false, 4);
    test_ij_4(false, false, 17);
    test_ij_4(true, false, 1);
    test_ij_4(true, false, 4);
    test_ij_4(true, false, 17);
}


/** \test \f$ b_{ij} = b_{ij} * a_{ij} \f$
 **/
void diag_tod_mult1_test::test_ij_1(bool recip, bool zero, size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_mult1_test::test_ij_1(" << recip << ", " << zero << ", "
        << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(0);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> va(sza, 0.0), vb(szb, 0.0);
        for(size_t i = 0; i < sza; i++) va[i] = drand48();
        for(size_t i = 0; i < szb; i++) vb[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza; i++) pa[i] = va[i];
            ca.ret_dataptr(ssa1, pa);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb; i++) pb[i] = vb[i];
            cb.ret_dataptr(ssb1, pb);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb_ref);

        diag_tod_mult1<2>(dta, recip).perform(zero, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_mult1<2>(ta, recip).perform(zero, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test \f$ b_{ij} = b_{ij} * a_{ij} \f$ (two subspaces in A: ij and ii)
 **/
void diag_tod_mult1_test::test_ij_2(bool recip, bool zero, size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_mult1_test::test_ij_2(" << recip << ", " << zero << ", "
        << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza1 = ni * ni;
        size_t sza2 = ni;
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(1), dtssa2(1);
        diag_tensor_subspace<2> dtssb1(0);
        dtssa1.set_diag_mask(0, m01);
        dtssa2.set_diag_mask(0, m11);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssa2 = dtsa.add_subspace(dtssa2);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> va1(sza1, 0.0), va2(sza2, 0.0), vb(szb, 0.0);
        for(size_t i = 0; i < sza1; i++) va1[i] = drand48();
        for(size_t i = 0; i < sza2; i++) va2[i] = drand48();
        for(size_t i = 0; i < szb; i++) vb[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            double *pa2 = ca.req_dataptr(ssa2);
            for(size_t i = 0; i < sza1; i++) pa1[i] = va1[i];
            for(size_t i = 0; i < sza2; i++) pa2[i] = va2[i];
            ca.ret_dataptr(ssa1, pa1);
            ca.ret_dataptr(ssa2, pa2);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb; i++) pb[i] = vb[i];
            cb.ret_dataptr(ssb1, pb);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb_ref);

        diag_tod_mult1<2>(dta, recip).perform(zero, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_mult1<2>(ta, recip).perform(zero, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test \f$ b_{ij} = b_{ij} * a_{ij} \f$ (two subspaces in both A and B:
        ij and ii)
 **/
void diag_tod_mult1_test::test_ij_3(bool recip, bool zero, size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_mult1_test::test_ij_3(" << recip << ", " << zero << ", "
        << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza1 = ni * ni;
        size_t sza2 = ni;
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb1 = ni;
        size_t szb2 = ni * ni;

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(1), dtssa2(1);
        diag_tensor_subspace<2> dtssb1(1), dtssb2(0);
        dtssa1.set_diag_mask(0, m01);
        dtssa2.set_diag_mask(0, m11);
        dtssb1.set_diag_mask(0, m11);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssa2 = dtsa.add_subspace(dtssa2);
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t ssb2 = dtsb.add_subspace(dtssb2);

        std::vector<double> va1(sza1, 0.0), va2(sza2, 0.0), vb1(szb1, 0.0),
            vb2(szb2, 0.0);
        for(size_t i = 0; i < sza1; i++) va1[i] = drand48();
        for(size_t i = 0; i < sza2; i++) va2[i] = drand48();
        for(size_t i = 0; i < szb1; i++) vb1[i] = drand48();
        for(size_t i = 0; i < szb2; i++) vb2[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            double *pa2 = ca.req_dataptr(ssa2);
            for(size_t i = 0; i < sza1; i++) pa1[i] = va1[i];
            for(size_t i = 0; i < sza2; i++) pa2[i] = va2[i];
            ca.ret_dataptr(ssa1, pa1);
            ca.ret_dataptr(ssa2, pa2);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            double *pb2 = cb.req_dataptr(ssb2);
            for(size_t i = 0; i < szb1; i++) pb1[i] = vb1[i];
            for(size_t i = 0; i < szb2; i++) pb2[i] = vb2[i];
            cb.ret_dataptr(ssb1, pb1);
            cb.ret_dataptr(ssb2, pb2);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb_ref);

        diag_tod_mult1<2>(dta, recip).perform(zero, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_mult1<2>(ta, recip).perform(zero, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test \f$ b_{ij} = b_{ij} * a_{ij} \f$ (only diagonals allowed in both A
        and B)
 **/
void diag_tod_mult1_test::test_ij_4(bool recip, bool zero, size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_mult1_test::test_ij_4(" << recip << ", " << zero << ", "
        << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza1 = ni;
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb1 = ni;

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(1);
        diag_tensor_subspace<2> dtssb1(1);
        dtssa1.set_diag_mask(0, m11);
        dtssb1.set_diag_mask(0, m11);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> va1(sza1, 0.0), vb1(szb1, 0.0);
        for(size_t i = 0; i < sza1; i++) va1[i] = drand48();
        for(size_t i = 0; i < szb1; i++) vb1[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza1; i++) pa1[i] = va1[i];
            ca.ret_dataptr(ssa1, pa1);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb1; i++) pb1[i] = vb1[i];
            cb.ret_dataptr(ssb1, pb1);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb_ref);

        diag_tod_mult1<2>(dta, recip).perform(zero, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_mult1<2>(ta, recip).perform(zero, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

