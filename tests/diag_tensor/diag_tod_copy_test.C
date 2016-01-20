#include <cstdlib>
#include <sstream>
#include <vector>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_import_raw.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_copy.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_copy_test.h"

namespace libtensor {


void diag_tod_copy_test::perform() throw(libtest::test_exception) {

    test_ij_1(1, 1, 1.0);
    test_ij_1(1, 4, 2.3);
    test_ij_1(4, 1, -0.5);
    test_ij_1(17, 17, -1.0);
    test_ij_2(1, 1, 1.0);
    test_ij_2(1, 4, 2.3);
    test_ij_2(4, 1, -0.5);
    test_ij_2(17, 17, -1.0);
    test_ij_3(1, 1.0);
    test_ij_3(4, -0.5);
    test_ij_3(17, -1.0);
}


/** \test \f$ b_{ij} = d a_{ij} \f$
 **/
void diag_tod_copy_test::test_ij_1(size_t ni, size_t nj, double d) {

    std::ostringstream tnss;
    tnss << "diag_tod_copy_test::test_ij_1(" << ni << ", " << nj << ", "
        << d << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> da(sza, 0.0), db(szb, 0.0);
        for(size_t i = 0; i < sza; i++) da[i] = drand48();
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza; i++) pa[i] = da[i];
            ca.ret_dataptr(ssa1, pa);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb; i++) pb[i] = db[i];
            cb.ret_dataptr(ssb1, pb);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);

        diag_tod_copy<2>(dta, d).perform(true, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_copy<2>(ta, d).perform(true, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test \f$ b_{ij} = d a_{ji} \f$
 **/
void diag_tod_copy_test::test_ij_2(size_t ni, size_t nj, double d) {

    std::ostringstream tnss;
    tnss << "diag_tod_copy_test::test_ij_2(" << ni << ", " << nj << ", "
        << d << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = nj - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        permutation<2> perm10;
        perm10.permute(0, 1);

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> da(sza, 0.0), db(szb, 0.0);
        for(size_t i = 0; i < sza; i++) da[i] = drand48();
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza; i++) pa[i] = da[i];
            ca.ret_dataptr(ssa1, pa);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb; i++) pb[i] = db[i];
            cb.ret_dataptr(ssb1, pb);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);

        diag_tod_copy<2>(dta, perm10, d).perform(true, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_copy<2>(ta, perm10, d).perform(true, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test \f$ b_{ij} = d a_{ij} \f$ (two subspaces in a)
 **/
void diag_tod_copy_test::test_ij_3(size_t ni, double d) {

    std::ostringstream tnss;
    tnss << "diag_tod_copy_test::test_ij_3(" << ni << ", " << d << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza1 = dimsa.get_size(), sza2 = ni;
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(0), dtssa2(1);
        diag_tensor_subspace<2> dtssb1(2);
        dtssa2.set_diag_mask(0, m11);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssa2 = dtsa.add_subspace(dtssa2);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> da1(sza1, 0.0), da2(sza2, 0.0), db(szb, 0.0);
        for(size_t i = 0; i < sza1; i++) da1[i] = drand48();
        for(size_t i = 0; i < sza2; i++) da2[i] = drand48();
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            double *pa2 = ca.req_dataptr(ssa2);
            for(size_t i = 0; i < sza1; i++) pa1[i] = da1[i];
            for(size_t i = 0; i < sza2; i++) pa2[i] = da2[i];
            ca.ret_dataptr(ssa1, pa1);
            ca.ret_dataptr(ssa2, pa2);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb; i++) pb[i] = db[i];
            cb.ret_dataptr(ssb1, pb);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb), tb_ref(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);

        diag_tod_copy<2>(dta, d).perform(true, dtb);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        tod_copy<2>(ta, d).perform(true, tb_ref);

        compare_ref<2>::compare(tn.c_str(), tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

