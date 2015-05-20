#include <cstdlib>
#include <sstream>
#include <vector>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_import_raw.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/diag_tensor/impl/diag_tod_contract2_part.h>
#include "../compare_ref.h"
#include "diag_tod_contract2_part_test.h"

namespace libtensor {


void diag_tod_contract2_part_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

    test_ij_ik_kj(1, 1, 1);
    test_ij_ik_kj(1, 1, 4);
    test_ij_ik_kj(4, 1, 1);
    test_ij_ik_kj(1, 4, 1);
    test_ij_ik_kj(4, 4, 4);
    test_ij_ik_kj(10, 11, 12);
    test_ij_ii_ij(1, 1);
    test_ij_ii_ij(1, 4);
    test_ij_ii_ij(4, 1);
    test_ij_ii_ij(4, 4);
    test_ij_ii_ij(10, 11);
    test_ii_ii_ij(1);
    test_ii_ii_ij(4);
    test_ii_ii_ij(10);
    test_ii_ii_ii(1);
    test_ii_ii_ii(4);
    test_ii_ii_ii(10);

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void diag_tod_contract2_part_test::test_ij_ik_kj(size_t ni, size_t nj,
    size_t nk) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_part_test::test_ij_ik_kj(" << ni << ", "
        << nj << ", " << nk << ")";
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

        diag_tensor_subspace<2> dtssa(0);
        diag_tensor_subspace<2> dtssb(2);
        dtssb.set_diag_mask(0, m01);
        dtssb.set_diag_mask(1, m10);
        diag_tensor_subspace<2> dtssc(1);
        dtssc.set_diag_mask(0, m10);

        std::vector<double> da(sza, 0.0), db(szb, 0.0), dc(szc, 0.0);
        for(size_t i = 0; i < sza; i++) da[i] = drand48();
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_import_raw<2>(&da[0], dimsa, index_range<2>(ia1, ia2)).perform(ta);
        tod_import_raw<2>(&db[0], dimsb, index_range<2>(ib1, ib2)).perform(tb);
        tod_set<2>().perform(true, tc);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        diag_tod_contract2_part<1, 1, 1>(contr, dtssa, dimsa, &da[0], dtssb,
            dimsb, &db[0]).perform(dtssc, dimsc, &dc[0], 1.0);
        tod_import_raw<2>(&dc[0], dimsc, index_range<2>(ic1, ic2)).perform(tc);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_part_test::test_ij_ii_ij(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_part_test::test_ij_ii_ij(" << ni << ", "
        << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<1> ria1, ria2;
        ria2[0] = ni - 1;
        dimensions<1> rdimsa(index_range<1>(ria1, ria2));
        size_t rsza = rdimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = nj - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa(1);
        dtssa.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb(0);
        diag_tensor_subspace<2> dtssc(1);
        dtssc.set_diag_mask(0, m10);

        std::vector<double> da(sza, 0.0), rda(rsza, 0.0), db(szb, 0.0),
            dc(szc, 0.0);
        for(size_t i = 0; i < ni; i++) {
            size_t ia = i * ni + i;
            size_t ria = i;
            da[ia] = rda[ria] = drand48();
        }
        for(size_t i = 0; i < szb; i++) db[i] = drand48();

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_import_raw<2>(&da[0], dimsa, index_range<2>(ia1, ia2)).perform(ta);
        tod_import_raw<2>(&db[0], dimsb, index_range<2>(ib1, ib2)).perform(tb);
        tod_set<2>().perform(true, tc);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        diag_tod_contract2_part<1, 1, 1>(contr, dtssa, dimsa, &rda[0], dtssb,
            dimsb, &db[0]).perform(dtssc, dimsc, &dc[0], 1.0);
        tod_import_raw<2>(&dc[0], dimsc, index_range<2>(ic1, ic2)).perform(tc);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_part_test::test_ii_ii_ij(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_part_test::test_ii_ii_ij(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<1> ria1, ria2;
        ria2[0] = ni - 1;
        dimensions<1> rdimsa(index_range<1>(ria1, ria2));
        size_t rsza = rdimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<1> rib1, rib2;
        rib2[0] = ni - 1;
        dimensions<1> rdimsb(index_range<1>(rib1, rib2));
        size_t rszb = rdimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = ni - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();
        index<1> ric1, ric2;
        ric2[0] = ni - 1;
        dimensions<1> rdimsc(index_range<1>(ric1, ric2));
        size_t rszc = rdimsc.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa(1);
        dtssa.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb(1);
        dtssb.set_diag_mask(0, m01);
        diag_tensor_subspace<2> dtssc(1);
        dtssc.set_diag_mask(0, m11);

        std::vector<double> da(sza, 0.0), rda(rsza, 0.0), db(szb, 0.0),
            dc(szc, 0.0), rdc(rszc, 0.0);
        for(size_t i = 0; i < ni; i++) {
            size_t ia = i * ni + i;
            size_t ria = i;
            da[ia] = rda[ria] = drand48();
        }
        for(size_t i = 0; i < ni; i++)
        for(size_t j = 0; j < ni; j++) {
            size_t ib = i * ni + j;
            db[ib] = drand48();
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_import_raw<2>(&da[0], dimsa, index_range<2>(ia1, ia2)).perform(ta);
        tod_import_raw<2>(&db[0], dimsb, index_range<2>(ib1, ib2)).perform(tb);
        tod_set<2>().perform(true, tc);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);
        {
            dense_tensor_wr_ctrl<2, double> cc_ref(tc_ref);
            double *pc = cc_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                if(i != j) pc[i * ni + j] = 0.0;
            }
            cc_ref.ret_dataptr(pc);
        }

        diag_tod_contract2_part<1, 1, 1>(contr, dtssa, dimsa, &rda[0], dtssb,
            dimsb, &db[0]).perform(dtssc, dimsc, &rdc[0], 1.0);
        for(size_t i = 0; i < ni; i++) {
            size_t ic = i * ni + i;
            size_t ric = i;
            dc[ic] = rdc[ric];
        }
        tod_import_raw<2>(&dc[0], dimsc, index_range<2>(ic1, ic2)).perform(tc);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_contract2_part_test::test_ii_ii_ii(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_contract2_part_test::test_ii_ii_ii(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = ni - 1;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        size_t sza = dimsa.get_size();
        index<1> ria1, ria2;
        ria2[0] = ni - 1;
        dimensions<1> rdimsa(index_range<1>(ria1, ria2));
        size_t rsza = rdimsa.get_size();
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = ni - 1;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        size_t szb = dimsb.get_size();
        index<1> rib1, rib2;
        rib2[0] = ni - 1;
        dimensions<1> rdimsb(index_range<1>(rib1, rib2));
        size_t rszb = rdimsb.get_size();
        index<2> ic1, ic2;
        ic2[0] = ni - 1; ic2[1] = ni - 1;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));
        size_t szc = dimsc.get_size();
        index<1> ric1, ric2;
        ric2[0] = ni - 1;
        dimensions<1> rdimsc(index_range<1>(ric1, ric2));
        size_t rszc = rdimsc.get_size();

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;

        diag_tensor_subspace<2> dtssa(1);
        dtssa.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssb(1);
        dtssb.set_diag_mask(0, m11);
        diag_tensor_subspace<2> dtssc(1);
        dtssc.set_diag_mask(0, m11);

        std::vector<double> da(sza, 0.0), rda(rsza, 0.0), db(szb, 0.0),
            rdb(rszb, 0.0), dc(szc, 0.0), rdc(rszc, 0.0);
        for(size_t i = 0; i < ni; i++) {
            size_t ia = i * ni + i;
            size_t ria = i;
            da[ia] = rda[ria] = drand48();
        }
        for(size_t i = 0; i < ni; i++) {
            size_t ib = i * ni + i;
            size_t rib = i;
            db[ib] = rdb[rib] = drand48();
        }

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_import_raw<2>(&da[0], dimsa, index_range<2>(ia1, ia2)).perform(ta);
        tod_import_raw<2>(&db[0], dimsb, index_range<2>(ib1, ib2)).perform(tb);
        tod_set<2>().perform(true, tc);

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        diag_tod_contract2_part<1, 1, 1>(contr, dtssa, dimsa, &rda[0], dtssb,
            dimsb, &rdb[0]).perform(dtssc, dimsc, &rdc[0], 1.0);
        for(size_t i = 0; i < ni; i++) {
            size_t ic = i * ni + i;
            size_t ric = i;
            dc[ic] = rdc[ric];
        }
        tod_import_raw<2>(&dc[0], dimsc, index_range<2>(ic1, ic2)).perform(tc);

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

