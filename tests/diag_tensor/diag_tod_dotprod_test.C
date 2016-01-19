#include <cstdlib>
#include <sstream>
#include <vector>
#include <libtensor/core/allocator.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_dotprod.h>
#include <libtensor/dense_tensor/tod_import_raw.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_dotprod.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_dotprod_test.h"

namespace libtensor {


void diag_tod_dotprod_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);
    linalg::rng_setup(0);

    try {

    test_1(1, 1);
    test_1(1, 4);
    test_1(4, 1);
    test_1(4, 4);
    test_1(10, 11);
    test_2(1, 1);
    test_2(1, 4);
    test_2(4, 1);
    test_2(4, 4);
    test_2(10, 11);
    test_3(1);
    test_3(4);
    test_3(10);
    test_3(17);
    test_4(1, 1);
    test_4(1, 4);
    test_4(4, 1);
    test_4(4, 4);
    test_4(10, 11);
    test_4(17, 16);

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


/** \f$\test \sum_{ij} a_{ij} b_{ij} \f$
 **/
void diag_tod_dotprod_test::test_1(size_t ni, size_t nj) {

    std::ostringstream tnss;
    tnss << "diag_tod_dotprod_test::test_1(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t sza1 = ni * nj;
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t szb1 = ni * nj;

        std::vector<double> da1(sza1, 0.0), db1(szb1, 0.0);
        linalg::rng_set_i_x(0, sza1, &da1[0], 1, 1.0);
        linalg::rng_set_i_x(0, szb1, &db1[0], 1, 1.0);

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza1; i++) pa1[i] = da1[i];
            ca.ret_dataptr(ssa1, pa1);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb1; i++) pb1[i] = db1[i];
            cb.ret_dataptr(ssb1, pb1);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        permutation<2> perma, permb;
        perma.permute(0, 1);
        permb.permute(0, 1);
        tensor_transf<2, double> tra(perma, scalar_transf<double>(-1.0)),
            trb(permb, scalar_transf<double>(-1.0));

        double d1 = diag_tod_dotprod<2>(dta, dtb).calculate();
        double d2 = diag_tod_dotprod<2>(dta, perma, dtb, permb).calculate();
        double d3 = diag_tod_dotprod<2>(dta, tra, dtb, trb).calculate();
        double d_ref = tod_dotprod<2>(ta, tb).calculate();

        if(fabs(d1 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << d1 << " (result), "
                << d_ref << " (reference), " << (d1 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d2 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << d2 << " (result), "
                << d_ref << " (reference), " << (d2 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d3 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << d3 << " (result), "
                << d_ref << " (reference), " << (d3 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \f$\test \sum_{ij} a_{ji} b_{ij} \f$
 **/
void diag_tod_dotprod_test::test_2(size_t ni, size_t nj) {

    std::ostringstream tnss;
    tnss << "diag_tod_dotprod_test::test_2(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = nj - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t sza1 = ni * nj;
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t szb1 = ni * nj;

        std::vector<double> da1(sza1, 0.0), db1(szb1, 0.0);
        linalg::rng_set_i_x(0, sza1, &da1[0], 1, 1.0);
        linalg::rng_set_i_x(0, szb1, &db1[0], 1, 1.0);

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza1; i++) pa1[i] = da1[i];
            ca.ret_dataptr(ssa1, pa1);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb1; i++) pb1[i] = db1[i];
            cb.ret_dataptr(ssb1, pb1);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        permutation<2> perma1, permb1, perma2, permb2;
        perma1.permute(0, 1);
        permb2.permute(0, 1);
        tensor_transf<2, double> tra(perma1, scalar_transf<double>(-1.0)),
            trb(permb1, scalar_transf<double>(-1.0));

        double d1 = diag_tod_dotprod<2>(dta, perma1, dtb, permb1).calculate();
        double d2 = diag_tod_dotprod<2>(dta, perma2, dtb, permb2).calculate();
        double d3 = diag_tod_dotprod<2>(dta, tra, dtb, trb).calculate();
        double d_ref = tod_dotprod<2>(ta, perma1, tb, permb1).calculate();

        if(fabs(d1 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << d1 << " (result), "
                << d_ref << " (reference), " << (d1 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d2 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << d2 << " (result), "
                << d_ref << " (reference), " << (d2 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d3 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << d3 << " (result), "
                << d_ref << " (reference), " << (d3 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \f$\test \sum_{i} a_{ii} b_{ij}\Delta_{ij} \f$
 **/
void diag_tod_dotprod_test::test_3(size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_dotprod_test::test_3(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(1);
        dtssa1.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t sza1 = ni;
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t szb1 = ni * ni;

        std::vector<double> da1(sza1, 0.0), db1(szb1, 0.0);
        linalg::rng_set_i_x(0, sza1, &da1[0], 1, 1.0);
        linalg::rng_set_i_x(0, szb1, &db1[0], 1, 1.0);

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            for(size_t i = 0; i < sza1; i++) pa1[i] = da1[i];
            ca.ret_dataptr(ssa1, pa1);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            for(size_t i = 0; i < szb1; i++) pb1[i] = db1[i];
            cb.ret_dataptr(ssb1, pb1);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        permutation<2> perma1, permb1, perma2, permb2;
        perma1.permute(0, 1);
        permb2.permute(0, 1);
        tensor_transf<2, double> tra(perma1, scalar_transf<double>(-1.0)),
            trb(permb1, scalar_transf<double>(-1.0));

        double d1 = diag_tod_dotprod<2>(dta, dtb).calculate();
        double d2 = diag_tod_dotprod<2>(dta, perma1, dtb, permb1).calculate();
        double d3 = diag_tod_dotprod<2>(dta, perma2, dtb, permb2).calculate();
        double d4 = diag_tod_dotprod<2>(dta, tra, dtb, trb).calculate();
        double d_ref = tod_dotprod<2>(ta, tb).calculate();

        if(fabs(d1 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << d1 << " (result), "
                << d_ref << " (reference), " << (d1 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d2 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << d2 << " (result), "
                << d_ref << " (reference), " << (d2 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d3 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << d3 << " (result), "
                << d_ref << " (reference), " << (d3 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d4 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << d4 << " (result), "
                << d_ref << " (reference), " << (d4 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \f$\test \sum_{i} a_{iijj} b_{ijij} \f$
 **/
void diag_tod_dotprod_test::test_4(size_t ni, size_t nj) {

    std::ostringstream tnss;
    tnss << "diag_tod_dotprod_test::test_4(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<4> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1; ia2[2] = nj - 1; ia2[3] = nj - 1;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        index<4> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1; ib2[2] = ni - 1; ib2[3] = nj - 1;
        dimensions<4> dimsb(index_range<4>(ib1, ib2));

        mask<4> m0011, m0101, m1010, m1100;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;

        diag_tensor_subspace<4> dtssa1(2);
        dtssa1.set_diag_mask(0, m1100);
        dtssa1.set_diag_mask(1, m0011);
        diag_tensor_subspace<4> dtssa2(1);
        dtssa2.set_diag_mask(0, m1100);
        diag_tensor_subspace<4> dtssa3(0);
        diag_tensor_subspace<4> dtssb1(2);
        dtssb1.set_diag_mask(0, m0101);
        dtssb1.set_diag_mask(1, m1010);
        diag_tensor_subspace<4> dtssb2(1);
        dtssb2.set_diag_mask(0, m0101);
        diag_tensor_subspace<4> dtssb3(1);
        dtssb3.set_diag_mask(0, m1010);

        diag_tensor_space<4> dtsa(dimsa);
        diag_tensor_space<4> dtsb(dimsb);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t sza1 = ni * nj;
        size_t ssa2 = dtsa.add_subspace(dtssa2);
        size_t sza2 = ni * nj * nj;
        size_t ssa3 = dtsa.add_subspace(dtssa3);
        size_t sza3 = ni * ni * nj * nj;
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t szb1 = ni * nj;
        size_t ssb2 = dtsb.add_subspace(dtssb2);
        size_t szb2 = ni * ni * nj;
        size_t ssb3 = dtsb.add_subspace(dtssb3);
        size_t szb3 = ni * nj * nj;

        std::vector<double> da1(sza1, 0.0), da2(sza2, 0.0), da3(sza3, 0.0),
            db1(szb1, 0.0), db2(szb2, 0.0), db3(szb3, 0.0);
        linalg::rng_set_i_x(0, sza1, &da1[0], 1, 1.0);
        linalg::rng_set_i_x(0, sza2, &da2[0], 1, 1.0);
        linalg::rng_set_i_x(0, sza3, &da3[0], 1, 1.0);
        linalg::rng_set_i_x(0, szb1, &db1[0], 1, 1.0);
        linalg::rng_set_i_x(0, szb2, &db2[0], 1, 1.0);
        linalg::rng_set_i_x(0, szb3, &db3[0], 1, 1.0);

        diag_tensor<4, double, allocator_t> dta(dtsa);
        diag_tensor<4, double, allocator_t> dtb(dtsb);

        {
            diag_tensor_wr_ctrl<4, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            double *pa2 = ca.req_dataptr(ssa2);
            double *pa3 = ca.req_dataptr(ssa3);
            for(size_t i = 0; i < sza1; i++) pa1[i] = da1[i];
            for(size_t i = 0; i < sza2; i++) pa2[i] = da2[i];
            for(size_t i = 0; i < sza3; i++) pa3[i] = da3[i];
            ca.ret_dataptr(ssa1, pa1);
            ca.ret_dataptr(ssa2, pa2);
            ca.ret_dataptr(ssa3, pa3);
        }
        {
            diag_tensor_wr_ctrl<4, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            double *pb2 = cb.req_dataptr(ssb2);
            double *pb3 = cb.req_dataptr(ssb3);
            for(size_t i = 0; i < szb1; i++) pb1[i] = db1[i];
            for(size_t i = 0; i < szb2; i++) pb2[i] = db2[i];
            for(size_t i = 0; i < szb3; i++) pb3[i] = db3[i];
            cb.ret_dataptr(ssb1, pb1);
            cb.ret_dataptr(ssb2, pb2);
            cb.ret_dataptr(ssb3, pb3);
        }

        dense_tensor<4, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb);

        tod_conv_diag_tensor<4>(dta).perform(ta);
        tod_conv_diag_tensor<4>(dtb).perform(tb);

        permutation<4> perma1, permb1, perma2, permb2, perma3, permb3;
        perma1.permute(1, 2);
        permb2.permute(1, 2);
        perma3.permute(1, 2).permute(2, 3);
        permb3.permute(2, 3);
        tensor_transf<4, double> tra(perma1, scalar_transf<double>(-1.0)),
            trb(permb1, scalar_transf<double>(1.0));

        double d1 = diag_tod_dotprod<4>(dta, perma1, dtb, permb1).calculate();
        double d2 = diag_tod_dotprod<4>(dta, perma2, dtb, permb2).calculate();
        double d3 = diag_tod_dotprod<4>(dta, perma3, dtb, permb3).calculate();
        double d4 = -diag_tod_dotprod<4>(dta, tra, dtb, trb).calculate();
        double d_ref = tod_dotprod<4>(ta, perma1, tb, permb1).calculate();

        if(fabs(d1 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << d1 << " (result), "
                << d_ref << " (reference), " << (d1 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d2 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << d2 << " (result), "
                << d_ref << " (reference), " << (d2 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d3 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << d3 << " (result), "
                << d_ref << " (reference), " << (d3 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

        if(fabs(d4 - d_ref) > 1e-14 * fabs(d_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << d4 << " (result), "
                << d_ref << " (reference), " << (d4 - d_ref) << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

