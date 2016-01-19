#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_dirsum.h>
#include "../compare_ref.h"
#include "tod_dirsum_test.h"

namespace libtensor {


void tod_dirsum_test::perform() throw(libtest::test_exception) {

    test_ij_i_j_1(1, 1);
    test_ij_i_j_1(2, 2);
    test_ij_i_j_1(3, 5);
    test_ij_i_j_1(16, 16);
    test_ij_i_j_1(1, 1, -0.5);
    test_ij_i_j_1(2, 2, 2.0);
    test_ij_i_j_1(3, 5, -1.0);
    test_ij_i_j_1(16, 16, 0.7);

    test_ij_i_j_2(1, 1);
    test_ij_i_j_2(2, 2);
    test_ij_i_j_2(3, 5);
    test_ij_i_j_2(16, 16);
    test_ij_i_j_2(1, 1, -0.5);
    test_ij_i_j_2(2, 2, 2.0);
    test_ij_i_j_2(3, 5, -1.0);
    test_ij_i_j_2(16, 16, 0.7);

    test_ikj_ij_k_1(1, 1, 1);
    test_ikj_ij_k_1(1, 1, 2);
    test_ikj_ij_k_1(1, 2, 1);
    test_ikj_ij_k_1(2, 1, 1);
    test_ikj_ij_k_1(1, 3, 3);
    test_ikj_ij_k_1(3, 3, 3);
    test_ikj_ij_k_1(3, 5, 7);
    test_ikj_ij_k_1(16, 16, 16);
    test_ikj_ij_k_1(1, 1, 1, -0.5);
    test_ikj_ij_k_1(1, 1, 2, 2.0);
    test_ikj_ij_k_1(1, 2, 1, -1.0);
    test_ikj_ij_k_1(2, 1, 1, 0.7);
    test_ikj_ij_k_1(1, 3, 3, 1.4);
    test_ikj_ij_k_1(3, 3, 3, 1.0);
    test_ikj_ij_k_1(3, 5, 7, -3.4);
    test_ikj_ij_k_1(16, 16, 16, 0.6);

    test_ikjl_ij_kl_1(1, 1, 1, 1);
    test_ikjl_ij_kl_1(1, 1, 1, 2);
    test_ikjl_ij_kl_1(1, 2, 1, 2);
    test_ikjl_ij_kl_1(3, 3, 3, 3);
    test_ikjl_ij_kl_1(3, 5, 7, 9);
    test_ikjl_ij_kl_1(9, 9, 7, 7);
    test_ikjl_ij_kl_1(16, 16, 16, 16);
    test_ikjl_ij_kl_1(1, 1, 1, 1, -0.7);
    test_ikjl_ij_kl_1(1, 1, 1, 2, 1.4);
    test_ikjl_ij_kl_1(1, 2, 1, 2, 0.1);
    test_ikjl_ij_kl_1(3, 3, 3, 3, -2.0);
    test_ikjl_ij_kl_1(3, 5, 7, 9, -1.0);
    test_ikjl_ij_kl_1(16, 16, 16, 16, 0.6);

    test_iklj_ij_kl_1(1, 1, 1, 1);
    test_iklj_ij_kl_1(1, 1, 1, 2);
    test_iklj_ij_kl_1(1, 2, 1, 2);
    test_iklj_ij_kl_1(3, 3, 3, 3);
    test_iklj_ij_kl_1(3, 5, 7, 9);
    test_iklj_ij_kl_1(9, 9, 7, 7);
    test_iklj_ij_kl_1(16, 16, 16, 16);
    test_iklj_ij_kl_1(1, 1, 1, 1, -0.7);
    test_iklj_ij_kl_1(1, 1, 1, 2, 1.4);
    test_iklj_ij_kl_1(1, 2, 1, 2, 0.1);
    test_iklj_ij_kl_1(3, 3, 3, 3, -2.0);
    test_iklj_ij_kl_1(3, 5, 7, 9, -1.0);
    test_iklj_ij_kl_1(16, 16, 16, 16, 0.6);
}


void tod_dirsum_test::test_ij_i_j_1(size_t ni, size_t nj, double d)
    throw(libtest::test_exception) {

    //    c_{ij} = a_i + b_j

    std::stringstream tnss;
    tnss << "tod_dirsum_test::test_ij_i_j_1(" << ni << ", " << nj << ", "
        << d << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator;

    try {

    index<1> ia1, ia2; ia2[0] = ni - 1;
    index<1> ib1, ib2; ib2[0] = nj - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //    Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    //    Generate reference data

    index<1> ia; index<1> ib; index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        ia[0] = i;
        ib[0] = j;
        ic[0] = i; ic[1] = j;
        abs_index<1> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
            (dta[aa.get_abs_index()] + dtb[ab.get_abs_index()]);
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //    Invoke the direct sum routine

    if(d == 0.0) {
        tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(true, tc);
    } else {
        scalar_transf<double> s1(1.), sd(d);
        tensor_transf<2, double> trc(permutation<2>(), sd);
        tod_dirsum<1, 1>(ta, s1, tb, s1, trc).perform(false, tc);
    }

    //    Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void tod_dirsum_test::test_ij_i_j_2(size_t ni, size_t nj, double d)
    throw(libtest::test_exception) {

    //  c_{ij} = a_i - b_j

    std::stringstream tnss;
    tnss << "tod_dirsum_test::test_ij_i_j_2(" << ni << ", " << nj << ", "
        << d << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator;

    try {

    index<1> ia1, ia2; ia2[0] = ni - 1;
    index<1> ib1, ib2; ib2[0] = nj - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //  Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = 0.0;//drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    //  Generate reference data

    index<1> ia; index<1> ib; index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        ia[0] = i;
        ib[0] = j;
        ic[0] = i; ic[1] = j;
        abs_index<1> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
            (dta[aa.get_abs_index()] - dtb[ab.get_abs_index()]);
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //  Invoke the direct sum routine

    if(d == 0.0) {
        tod_dirsum<1, 1>(ta, 1.0, tb, -1.0).perform(true, tc);
    } else {
        scalar_transf<double> s1(1.), s2(-1.), sd(d);
        tensor_transf<2, double> trc(permutation<2>(), sd);
        tod_dirsum<1, 1>(ta, s1, tb, s2, trc).perform(false, tc);
    }

    //  Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void tod_dirsum_test::test_ikj_ij_k_1(size_t ni, size_t nj, size_t nk,
    double d) throw(libtest::test_exception) {

    // c_{ikj} = a_{ij} + b_k

    std::stringstream tnss;
    tnss << "tod_dirsum_test::test_ikj_ij_k(" << ni << ", " << nj << ", "
        << nk << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator;

    try {

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<1> ib1, ib2;
    ib2[0] = nk - 1;
    index<3> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //    Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    //    Generate reference data

    index<2> ia; index<1> ib; index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
        ia[0] = i; ia[1] = j;
        ib[0] = k;
        ic[0] = i; ic[1] = k; ic[2] = j;
        abs_index<2> aa(ia, dima);
        abs_index<1> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
            (dta[aa.get_abs_index()] + dtb[ab.get_abs_index()]);
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //    Invoke the direct sum routine

    permutation<3> permc;
    permc.permute(1, 2); // ijk -> ikj
    if(d == 0.0) {
        tod_dirsum<2, 1>(ta, 1.0, tb, 1.0, permc).perform(true, tc);
    } else {
        scalar_transf<double> s1(1.), sd(d);
        tensor_transf<3, double> trc(permc, sd);
        tod_dirsum<2, 1>(ta, s1, tb, s1, trc).perform(false, tc);
    }

    //    Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dirsum_test::test_ikjl_ij_kl_1(size_t ni, size_t nj, size_t nk,
    size_t nl, double d) throw(libtest::test_exception) {

    // c_{ikjl} = a_{ij} + b_{kl}

    std::stringstream tnss;
    tnss << "tod_dirsum_test::test_ikjl_ij_kl(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator;

    try {

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1; ic2[3] = nl - 1; 
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //    Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    //    Generate reference data

    index<2> ia; index<2> ib; index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
        ia[0] = i; ia[1] = j;
        ib[0] = k; ib[1] = l;
        ic[0] = i; ic[1] = k; ic[2] = j; ic[3] = l;
        abs_index<2> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
            (dta[aa.get_abs_index()] + dtb[ab.get_abs_index()]);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //    Invoke the direct sum routine

    permutation<4> permc;
    permc.permute(1, 2); // ijkl -> ikjl
    if(d == 0.0) {
        tod_dirsum<2, 2>(ta, 1.0, tb, 1.0, permc).perform(true, tc);
    } else {
        scalar_transf<double> s1(1.), sd(d);
        tensor_transf<4, double> trc(permc, sd);
        tod_dirsum<2, 2>(ta, s1, tb, s1, trc).perform(false, tc);
    }

    //    Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dirsum_test::test_iklj_ij_kl_1(size_t ni, size_t nj, size_t nk,
    size_t nl, double d) throw(libtest::test_exception) {

    // c_{iklj} = a_{ij} + b_{kl}

    std::stringstream tnss;
    tnss << "tod_dirsum_test::test_iklj_ij_kl(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator;

    try {

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nl - 1; ic2[3] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //    Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    //    Generate reference data

    index<2> ia; index<2> ib; index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
        ia[0] = i; ia[1] = j;
        ib[0] = k; ib[1] = l;
        ic[0] = i; ic[1] = k; ic[2] = l; ic[3] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
            (dta[aa.get_abs_index()] + dtb[ab.get_abs_index()]);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //    Invoke the direct sum routine

    permutation<4> permc;
    permc.permute(1, 2).permute(2, 3); // ijkl -> iklj
    if(d == 0.0) {
        tod_dirsum<2, 2>(ta, 1.0, tb, 1.0, permc).perform(true, tc);
    } else {
        scalar_transf<double> s1(1.), sd(d);
        tensor_transf<4, double> trc(permc, sd);
        tod_dirsum<2, 2>(ta, s1, tb, s1, trc).perform(false, tc);
    }

    //    Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
