#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_ewmult2.h>
#include "to_ewmult2_test.h"
#include "../compare_ref.h"

namespace libtensor {

void to_ewmult2_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_emult2_test<double>  ";
    to_ewmult2_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_emult2_test<float>  ";
    to_ewmult2_test_x<float> t_float;
    t_float.perform();
}

template<>
const double to_ewmult2_test_x<double>::k_thresh = 1e-14;
template<>
const float to_ewmult2_test_x<float>::k_thresh = 1e-5;

template<typename T>
void to_ewmult2_test_x<T>::perform() throw(libtest::test_exception) {

    test_i_i_i(1);
    test_i_i_i(6);
    test_i_i_i(16);
    test_i_i_i(17);
    test_i_i_i(1, -0.5);
    test_i_i_i(6, -2.0);
    test_i_i_i(16, 1.2);
    test_i_i_i(17, 0.7);

    test_ij_ij_ij(1, 1);
    test_ij_ij_ij(1, 6);
    test_ij_ij_ij(6, 1);
    test_ij_ij_ij(16, 16);
    test_ij_ij_ij(6, 17);
    test_ij_ij_ij(1, 1, -0.5);
    test_ij_ij_ij(1, 6, -2.0);
    test_ij_ij_ij(6, 1, 1.2);
    test_ij_ij_ij(16, 16, 0.7);
    test_ij_ij_ij(17, 6, -1.3);

    test_ij_ij_ji(1, 1);
    test_ij_ij_ji(1, 6);
    test_ij_ij_ji(6, 1);
    test_ij_ij_ji(16, 16);
    test_ij_ij_ji(6, 17);
    test_ij_ij_ji(1, 1, -0.5);
    test_ij_ij_ji(1, 6, -2.0);
    test_ij_ij_ji(6, 1, 1.2);
    test_ij_ij_ji(16, 16, 0.7);
    test_ij_ij_ji(17, 6, -1.3);

    test_ijk_jki_kij(1, 1, 1);
    test_ijk_jki_kij(1, 1, 6);
    test_ijk_jki_kij(6, 1, 1);
    test_ijk_jki_kij(6, 6, 1);
    test_ijk_jki_kij(16, 17, 16);
    test_ijk_jki_kij(1, 1, 1, -1.3);
    test_ijk_jki_kij(1, 1, 6, 0.7);
    test_ijk_jki_kij(6, 1, 1, 1.2);
    test_ijk_jki_kij(6, 6, 1, -2.0);
    test_ijk_jki_kij(16, 17, 16, -0.5);

    test_ijk_ik_kj(1, 1, 1);
    test_ijk_ik_kj(1, 1, 6);
    test_ijk_ik_kj(6, 1, 1);
    test_ijk_ik_kj(6, 6, 1);
    test_ijk_ik_kj(16, 17, 16);
    test_ijk_ik_kj(1, 1, 1, -1.3);
    test_ijk_ik_kj(1, 1, 6, 0.7);
    test_ijk_ik_kj(6, 1, 1, 1.2);
    test_ijk_ik_kj(6, 6, 1, -2.0);
    test_ijk_ik_kj(16, 17, 16, -0.5);

    test_ijkl_kj_ikl(1, 1, 1, 1);
    test_ijkl_kj_ikl(1, 6, 1, 1);
    test_ijkl_kj_ikl(1, 1, 6, 1);
    test_ijkl_kj_ikl(6, 1, 1, 6);
    test_ijkl_kj_ikl(6, 6, 1, 1);
    test_ijkl_kj_ikl(16, 17, 16, 15);
    test_ijkl_kj_ikl(1, 1, 1, 1, -0.5);
    test_ijkl_kj_ikl(1, 6, 1, 1, -2.0);
    test_ijkl_kj_ikl(1, 1, 6, 1, 1.2);
    test_ijkl_kj_ikl(6, 1, 1, 6, 0.7);
    test_ijkl_kj_ikl(6, 6, 1, 1, -1.3);
    test_ijkl_kj_ikl(16, 17, 16, 15, 1.0);

    test_ijkl_ljk_jil(1, 1, 1, 1);
    test_ijkl_ljk_jil(1, 6, 1, 1);
    test_ijkl_ljk_jil(1, 1, 6, 1);
    test_ijkl_ljk_jil(6, 1, 1, 6);
    test_ijkl_ljk_jil(6, 6, 1, 1);
    test_ijkl_ljk_jil(16, 17, 16, 15);
    test_ijkl_ljk_jil(1, 1, 1, 1, -0.5);
    test_ijkl_ljk_jil(1, 6, 1, 1, -2.0);
    test_ijkl_ljk_jil(1, 1, 6, 1, 1.2);
    test_ijkl_ljk_jil(6, 1, 1, 6, 0.7);
    test_ijkl_ljk_jil(6, 6, 1, 1, -1.3);
    test_ijkl_ljk_jil(16, 17, 16, 15, 1.0);
}


/** \test Tests $c_i = a_i b_i$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_i_i_i(size_t ni, T d)
    throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_i_i_i(" << ni << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<1> ia1, ia2; ia2[0] = ni - 1;
    index<1> ib1, ib2; ib2[0] = ni - 1;
    index<1> ic1, ic2; ic2[0] = ni - 1;
    dimensions<1> dimsa(index_range<1>(ia1, ia2));
    dimensions<1> dimsb(index_range<1>(ib1, ib2));
    dimensions<1> dimsc(index_range<1>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<1, T, allocator> ta(dimsa);
    dense_tensor<1, T, allocator> tb(dimsb);
    dense_tensor<1, T, allocator> tc(dimsc);
    dense_tensor<1, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;

    {
    dense_tensor_ctrl<1, T> tca(ta);
    dense_tensor_ctrl<1, T> tcb(tb);
    dense_tensor_ctrl<1, T> tcc(tc);
    dense_tensor_ctrl<1, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<1> ia; index<1> ib; index<1> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
        ia[0] = i;
        ib[0] = i;
        ic[0] = i;
        abs_index<1> aia(ia, dimsa), aib(ib, dimsb), aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    to_ewmult2<0, 0, 1, T> op(ta, tb, (d == 0 ? 1.0 : d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<1, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests $c_{ij} = a_{ij} b_{ij}$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_ij_ij_ij(size_t ni, size_t nj, T d)
    throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_ij_ij_ij(" << ni << ", " << nj << ", "
        << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = nj - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<2> dimsb(index_range<2>(ib1, ib2));
    dimensions<2> dimsc(index_range<2>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<2, T, allocator> ta(dimsa);
    dense_tensor<2, T, allocator> tb(dimsb);
    dense_tensor<2, T, allocator> tc(dimsc);
    dense_tensor<2, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;
    T d2 = drand48();

    {
    dense_tensor_ctrl<2, T> tca(ta);
    dense_tensor_ctrl<2, T> tcb(tb);
    dense_tensor_ctrl<2, T> tcc(tc);
    dense_tensor_ctrl<2, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<2> ia; index<2> ib; index<2> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        ia[0] = i; ia[1] = j;
        ib[0] = i; ib[1] = j;
        ic[0] = i; ic[1] = j;
        abs_index<2> aia(ia, dimsa), aib(ib, dimsb), aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 * d2 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    to_ewmult2<0, 0, 2, T> op(ta, tb, (d == 0.0 ? d2 : d2 * d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<2, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests $c_{ij} = a_{ij} b_{ji}$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_ij_ij_ji(size_t ni, size_t nj, T d)
    throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_ij_ij_ji(" << ni << ", " << nj << ", "
        << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2; ib2[0] = nj - 1; ib2[1] = ni - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<2> dimsb(index_range<2>(ib1, ib2));
    dimensions<2> dimsc(index_range<2>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<2, T, allocator> ta(dimsa);
    dense_tensor<2, T, allocator> tb(dimsb);
    dense_tensor<2, T, allocator> tc(dimsc);
    dense_tensor<2, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;
    T d2 = drand48();

    {
    dense_tensor_ctrl<2, T> tca(ta);
    dense_tensor_ctrl<2, T> tcb(tb);
    dense_tensor_ctrl<2, T> tcc(tc);
    dense_tensor_ctrl<2, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<2> ia; index<2> ib; index<2> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        ia[0] = i; ia[1] = j;
        ib[0] = j; ib[1] = i;
        ic[0] = i; ic[1] = j;
        abs_index<2> aia(ia, dimsa), aib(ib, dimsb), aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 * d2 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    permutation<2> perma;
    permutation<2> permb; permb.permute(0, 1);
    permutation<2> permc;
    to_ewmult2<0, 0, 2, T> op(ta, perma, tb, permb, permc, (d == 0 ? d2 : d2 * d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<2, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests $c_{ijk} = a_{jki} b_{kij}$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_ijk_jki_kij(size_t ni, size_t nj, size_t nk,
    T d) throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_ijk_jki_kij(" << ni << ", "
        << nj << ", " << nk << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<3> ia1, ia2; ia2[0] = nj - 1; ia2[1] = nk - 1; ia2[2] = ni - 1;
    index<3> ib1, ib2; ib2[0] = nk - 1; ib2[1] = ni - 1; ib2[2] = nj - 1;
    index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<3> dimsa(index_range<3>(ia1, ia2));
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    dimensions<3> dimsc(index_range<3>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<3, T, allocator> ta(dimsa);
    dense_tensor<3, T, allocator> tb(dimsb);
    dense_tensor<3, T, allocator> tc(dimsc);
    dense_tensor<3, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;
    T d2 = drand48();

    {
    dense_tensor_ctrl<3, T> tca(ta);
    dense_tensor_ctrl<3, T> tcb(tb);
    dense_tensor_ctrl<3, T> tcc(tc);
    dense_tensor_ctrl<3, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<3> ia; index<3> ib; index<3> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
        ia[0] = j; ia[1] = k; ia[2] = i;
        ib[0] = k; ib[1] = i; ib[2] = j;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<3> aia(ia, dimsa), aib(ib, dimsb), aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 * d2 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    permutation<3> perma; perma.permute(0, 1).permute(1, 2); // jki->kij
    permutation<3> permb; // kij
    permutation<3> permc; permc.permute(0, 1).permute(1, 2); // kij->ijk
    to_ewmult2<0, 0, 3, T> op(ta, perma, tb, permb, permc, (d == 0.0 ? d2 : d2 * d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<3, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests $c_{ijk} = a_{ik} b_{kj}$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_ijk_ik_kj(size_t ni, size_t nj, size_t nk,
    T d) throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_ijk_ik_kj(" << ni << ", "
        << nj << ", " << nk << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = nk - 1;
    index<2> ib1, ib2; ib2[0] = nk - 1; ib2[1] = nj - 1;
    index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<2> dimsb(index_range<2>(ib1, ib2));
    dimensions<3> dimsc(index_range<3>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<2, T, allocator> ta(dimsa);
    dense_tensor<2, T, allocator> tb(dimsb);
    dense_tensor<3, T, allocator> tc(dimsc);
    dense_tensor<3, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;
    T d2 = drand48();

    {
    dense_tensor_ctrl<2, T> tca(ta);
    dense_tensor_ctrl<2, T> tcb(tb);
    dense_tensor_ctrl<3, T> tcc(tc);
    dense_tensor_ctrl<3, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<2> ia; index<2> ib; index<3> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
        ia[0] = i; ia[1] = k;
        ib[0] = k; ib[1] = j;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<2> aia(ia, dimsa), aib(ib, dimsb);
        abs_index<3> aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 * d2 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    permutation<2> perma; // ik
    permutation<2> permb; permb.permute(0, 1); // kj->jk
    permutation<3> permc; // ijk
    to_ewmult2<1, 1, 1, T> op(ta, perma, tb, permb, permc, (d == 0.0 ? d2 : d2 * d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<3, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests $c_{ijkl} = a_{kj} b_{ikl}$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_ijkl_kj_ikl(size_t ni, size_t nj, size_t nk,
    size_t nl, T d) throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_ijkl_kj_ikl(" << ni << ", "
        << nj << ", " << nk << ", " << nl << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<2> ia1, ia2; ia2[0] = nk - 1; ia2[1] = nj - 1;
    index<3> ib1, ib2; ib2[0] = ni - 1; ib2[1] = nk - 1; ib2[2] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<2, T, allocator> ta(dimsa);
    dense_tensor<3, T, allocator> tb(dimsb);
    dense_tensor<4, T, allocator> tc(dimsc);
    dense_tensor<4, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;
    T d2 = drand48();

    {
    dense_tensor_ctrl<2, T> tca(ta);
    dense_tensor_ctrl<3, T> tcb(tb);
    dense_tensor_ctrl<4, T> tcc(tc);
    dense_tensor_ctrl<4, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<2> ia; index<3> ib; index<4> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
        ia[0] = k; ia[1] = j;
        ib[0] = i; ib[1] = k; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<2> aia(ia, dimsa);
        abs_index<3> aib(ib, dimsb);
        abs_index<4> aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 * d2 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    permutation<2> perma; perma.permute(0, 1); // kj->jk
    permutation<3> permb; permb.permute(1, 2); // ikl->ilk
    permutation<4> permc; permc.permute(0, 1).permute(2, 3); // jilk->ijkl
    to_ewmult2<1, 2, 1, T> op(ta, perma, tb, permb, permc, (d == 0.0 ? d2 : d2 * d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<4, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests $c_{ijkl} = a_{ljk} b_{jil}$
 **/
template<typename T>
void to_ewmult2_test_x<T>::test_ijkl_ljk_jil(size_t ni, size_t nj, size_t nk,
    size_t nl, T d) throw(libtest::test_exception) {

    std::stringstream tnss;
    tnss << "to_ewmult2_test_x<T>::test_ijkl_ljk_jil(" << ni << ", "
        << nj << ", " << nk << ", " << nl << ", " << d << ")";
    std::string tns = tnss.str();

    typedef allocator<T> allocator;

    try {

    index<3> ia1, ia2; ia2[0] = nl - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
    index<3> ib1, ib2; ib2[0] = nj - 1; ib2[1] = ni - 1; ib2[2] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dimsa(index_range<3>(ia1, ia2));
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    size_t sza = dimsa.get_size(), szb = dimsb.get_size(),
        szc = dimsc.get_size();

    dense_tensor<3, T, allocator> ta(dimsa);
    dense_tensor<3, T, allocator> tb(dimsb);
    dense_tensor<4, T, allocator> tc(dimsc);
    dense_tensor<4, T, allocator> tc_ref(dimsc);

    T cij_max = 0.0;
    T d2 = drand48();

    {
    dense_tensor_ctrl<3, T> tca(ta);
    dense_tensor_ctrl<3, T> tcb(tb);
    dense_tensor_ctrl<4, T> tcc(tc);
    dense_tensor_ctrl<4, T> tcc_ref(tc_ref);
    T *dta = tca.req_dataptr();
    T *dtb = tcb.req_dataptr();
    T *dtc1 = tcc.req_dataptr();
    T *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    index<3> ia; index<3> ib; index<4> ic;
    T d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
        ia[0] = l; ia[1] = j; ia[2] = k;
        ib[0] = j; ib[1] = i; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aia(ia, dimsa);
        abs_index<3> aib(ib, dimsb);
        abs_index<4> aic(ic, dimsc);
        dtc2[aic.get_abs_index()] += d1 * d2 *
        dta[aia.get_abs_index()] * dtb[aib.get_abs_index()];
        cij_max = std::max(cij_max, std::abs(dtc2[aic.get_abs_index()]));
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the routine

    permutation<3> perma; perma.permute(1, 2).permute(0, 1); // ljk->klj
    permutation<3> permb; permb.permute(0, 1).permute(1, 2); // jil->ilj
    permutation<4> permc;
    permc.permute(2, 3).permute(1, 2).permute(0, 2); // kilj->ijkl
    to_ewmult2<1, 1, 2, T> op(ta, perma, tb, permb, permc, (d == 0.0 ? d2 : d2 * d));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref_x<4, T>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
