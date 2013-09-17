#include <cstdlib>
#include <sstream>
#include <vector>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_import_raw.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_contract2.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_contract2_test.h"

namespace libtensor {


void diag_tod_contract2_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

    test_1_1_1_01(1, 1, 1);
    test_1_1_1_01(1, 1, 4);
    test_1_1_1_01(4, 1, 1);
    test_1_1_1_01(1, 4, 1);
    test_1_1_1_01(4, 4, 4);
    test_1_1_1_01(10, 11, 12);
    test_1_1_1_02(1, 1, 1);
    test_1_1_1_02(1, 1, 4);
    test_1_1_1_02(4, 1, 1);
    test_1_1_1_02(1, 4, 1);
    test_1_1_1_02(4, 4, 4);
    test_1_1_1_02(10, 11, 12);
    test_1_1_1_03(1, 1);
    test_1_1_1_03(1, 4);
    test_1_1_1_03(4, 1);
    test_1_1_1_03(4, 4);
    test_1_1_1_03(10, 11);
    test_1_1_1_04(1);
    test_1_1_1_04(4);
    test_1_1_1_04(10);
    test_1_1_1_05(1);
    test_1_1_1_05(4);
    test_1_1_1_05(10);

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void diag_tod_contract2_test::test_1_1_1_01(size_t ni, size_t nj, size_t nk) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_test::test_1_1_1_01(" << ni << ", " << nj
        << ", " << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nk - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = nk - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = nj - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);
        diag_tensor_subspace<2> dtssc1(1);
        dtssc1.set_diag_mask(0, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);
        diag_tensor_space<2> dtsc(dimsc);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t ssc1 = dtsc.add_subspace(dtssc1);

        std::vector<double> da(sza, 0.0), db(szb, 0.0), dc(szc, 0.0);
        for(size_t i = 0; i < sza; i++) da[i] = drand48();
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);
        diag_tensor<2, double, allocator_t> dtc(dtsc);

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
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        diag_tod_contract2<1, 1, 1>(contr, dta, dtb).perform(true, dtc);
        tod_conv_diag_tensor<2>(dtc).perform(tc);

        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_test::test_1_1_1_02(size_t ni, size_t nj, size_t nk) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_test::test_1_1_1_02(" << ni << ", " << nj
        << ", " << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nk - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = nk - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = nj - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();

        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        diag_tensor_subspace<2> dtssa1(0);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);
        diag_tensor_subspace<2> dtssc1(1), dtssc2(0);
        dtssc1.set_diag_mask(0, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);
        diag_tensor_space<2> dtsc(dimsc);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t ssc1 = dtsc.add_subspace(dtssc1);
        size_t ssc2 = dtsc.add_subspace(dtssc2);

        std::vector<double> da(sza, 0.0), db(szb, 0.0), dc(szc, 0.0);
        for(size_t i = 0; i < sza; i++) da[i] = drand48();
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);
        diag_tensor<2, double, allocator_t> dtc(dtsc);

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
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        diag_tod_contract2<1, 1, 1>(contr, dta, dtb).perform(true, dtc);
        tod_conv_diag_tensor<2>(dtc).perform(tc);

        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_test::test_1_1_1_03(size_t ni, size_t nj) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_test::test_1_1_1_03(" << ni << ", "
        << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = nj - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(0), dtssa2(1);
        dtssa2.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb1(2);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);
        diag_tensor_subspace<2> dtssc1(1), dtssc2(0);
        dtssc1.set_diag_mask(0, m10);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);
        diag_tensor_space<2> dtsc(dimsc);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssa2 = dtsa.add_subspace(dtssa2);
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t ssc1 = dtsc.add_subspace(dtssc1);
        size_t ssc2 = dtsc.add_subspace(dtssc2);

        std::vector<double> da1(ni * ni, 0.0), da2(ni, 0.0), db1(ni * nj, 0.0);
        for(size_t i = 0; i < ni * ni; i++) da1[i] = drand48();
        for(size_t i = 0; i < ni; i++) da2[i] = drand48();
        for(size_t i = 0; i < ni * nj; i++) db1[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);
        diag_tensor<2, double, allocator_t> dtc(dtsc);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            memcpy(pa1, &da1[0], sizeof(double) * ni * ni);
            ca.ret_dataptr(ssa1, pa1);
            double *pa2 = ca.req_dataptr(ssa2);
            memcpy(pa2, &da2[0], sizeof(double) * ni);
            ca.ret_dataptr(ssa2, pa2);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            memcpy(pb1, &db1[0], sizeof(double) * ni * nj);
            cb.ret_dataptr(ssb1, pb1);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        diag_tod_contract2<1, 1, 1>(contr, dta, dtb).perform(true, dtc);
        tod_conv_diag_tensor<2>(dtc).perform(tc);

        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_test::test_1_1_1_04(size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_test::test_1_1_1_04(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = ni - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(0), dtssa2(1);
        dtssa2.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb1(2), dtssb2(1);
        dtssb1.set_diag_mask(0, m01);
        dtssb1.set_diag_mask(1, m10);
        dtssb2.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssc1(1), dtssc2(1);
        dtssc1.set_diag_mask(0, m10);
        dtssc2.set_diag_mask(0, m11);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);
        diag_tensor_space<2> dtsc(dimsc);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssa2 = dtsa.add_subspace(dtssa2);
        size_t ssb1 = dtsb.add_subspace(dtssb1);
        size_t ssb2 = dtsb.add_subspace(dtssb2);
        size_t ssc1 = dtsc.add_subspace(dtssc1);
        size_t ssc2 = dtsc.add_subspace(dtssc2);

        std::vector<double> da1(ni * ni, 0.0), da2(ni, 0.0), db1(ni * ni, 0.0),
            db2(ni, 0.0);
        for(size_t i = 0; i < ni * ni; i++) da1[i] = drand48();
        for(size_t i = 0; i < ni; i++) da2[i] = drand48();
        for(size_t i = 0; i < ni * ni; i++) db1[i] = drand48();
        for(size_t i = 0; i < ni; i++) db2[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);
        diag_tensor<2, double, allocator_t> dtc(dtsc);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            memcpy(pa1, &da1[0], sizeof(double) * ni * ni);
            ca.ret_dataptr(ssa1, pa1);
            double *pa2 = ca.req_dataptr(ssa2);
            memcpy(pa2, &da2[0], sizeof(double) * ni);
            ca.ret_dataptr(ssa2, pa2);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            memcpy(pb1, &db1[0], sizeof(double) * ni * ni);
            cb.ret_dataptr(ssb1, pb1);
            double *pb2 = cb.req_dataptr(ssb2);
            memcpy(pb2, &db2[0], sizeof(double) * ni);
            cb.ret_dataptr(ssb2, pb2);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        diag_tod_contract2<1, 1, 1>(contr, dta, dtb).perform(true, dtc);
        tod_conv_diag_tensor<2>(dtc).perform(tc);

        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_test::test_1_1_1_05(size_t ni) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_test::test_1_1_1_05(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = ni - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa1(1);
        dtssa1.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb1(0);

        diag_tensor_space<2> dtsa(dimsa);
        diag_tensor_space<2> dtsb(dimsb);
        diag_tensor_space<2> dtsc(dimsc);

        size_t ssa1 = dtsa.add_subspace(dtssa1);
        size_t ssb1 = dtsb.add_subspace(dtssb1);

        std::vector<double> da1(ni, 0.0), db1(ni * ni, 0.0);
        for(size_t i = 0; i < ni; i++) da1[i] = drand48();
        for(size_t i = 0; i < ni * ni; i++) db1[i] = drand48();

        diag_tensor<2, double, allocator_t> dta(dtsa);
        diag_tensor<2, double, allocator_t> dtb(dtsb);
        diag_tensor<2, double, allocator_t> dtc(dtsc);

        {
            diag_tensor_wr_ctrl<2, double> ca(dta);
            double *pa1 = ca.req_dataptr(ssa1);
            memcpy(pa1, &da1[0], sizeof(double) * ni);
            ca.ret_dataptr(ssa1, pa1);
        }
        {
            diag_tensor_wr_ctrl<2, double> cb(dtb);
            double *pb1 = cb.req_dataptr(ssb1);
            memcpy(pb1, &db1[0], sizeof(double) * ni * ni);
            cb.ret_dataptr(ssb1, pb1);
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

        tod_conv_diag_tensor<2>(dta).perform(ta);
        tod_conv_diag_tensor<2>(dtb).perform(tb);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        diag_tod_contract2<1, 1, 1>(contr, dta, dtb).perform(true, dtc);
        tod_conv_diag_tensor<2>(dtc).perform(tc);

        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

