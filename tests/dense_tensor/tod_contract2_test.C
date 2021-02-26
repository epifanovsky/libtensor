#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_contract2.h>
#include "../compare_ref.h"
#include "../test_utils.h"

using namespace libtensor;
const double k_thresh = 5e-14;

#if 0
int test_0_p_p(size_t np, double d) {

    // c = \sum_p a_p b_p

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_0_p_p(" << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<1> ia1, ia2; ia2[0] = np - 1;
    libtensor::index<1> ib1, ib2; ib2[0] = np - 1;
    libtensor::index<0> ic1, ic2;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<0> dimc(index_range<0>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<0, double, allocator> tc(dimc);
    dense_tensor<0, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<0, double> tcc(tc);
    dense_tensor_ctrl<0, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<1> ia; libtensor::index<1> ib; libtensor::index<0> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t p = 0; p < np; p++) {
        ia[0] = p;
        ib[0] = p;
        abs_index<1> aa(ia, dima), ab(ib, dimb);
        dtc2[0] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    cij_max = fabs(dtc2[0]);

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<0, 0, 1> contr;
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<0, 0, 1, double>(contr, ta, tb).perform(true, tc);
    else to_contract2<0, 0, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<0>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}
#endif


int test_i_p_pi(size_t ni, size_t np, double d) {

    // c_i = \sum_p a_p b_{pi}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_i_p_pi(" << ni << ", " << np << ", "
        << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<1> ia1, ia2; ia2[0] = np - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
    libtensor::index<1> ic1, ic2; ic2[0] = ni - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<1> dimc(index_range<1>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<1, double, allocator> tc(dimc);
    dense_tensor<1, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<1, double> tcc(tc);
    dense_tensor_ctrl<1, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<1> ia; libtensor::index<2> ib; libtensor::index<1> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p;
        ib[0] = p; ib[1] = i;
        ic[0] = i;
        abs_index<1> aa(ia, dima), ac(ic, dimc);
        abs_index<2> ab(ib, dimb);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<0, 1, 1> contr;
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<0, 1, 1, double>(contr, ta, tb).perform(true, tc);
    else to_contract2<0, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_i_p_ip(size_t ni, size_t np, double d) {

    // c_i = \sum_p a_p b_{ip}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_i_p_ip(" << ni << ", " << np << ", "
        << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<1> ia1, ia2; ia2[0] = np - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
    libtensor::index<1> ic1, ic2; ic2[0] = ni - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<1> dimc(index_range<1>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<1, double, allocator> tc(dimc);
    dense_tensor<1, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<1, double> tcc(tc);
    dense_tensor_ctrl<1, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<1> ia; libtensor::index<2> ib; libtensor::index<1> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p;
        ib[0] = i; ib[1] = p;
        ic[0] = i;
        abs_index<1> aa(ia, dima), ac(ic, dimc);
        abs_index<2> ab(ib, dimb);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<0, 1, 1> contr;
    contr.contract(0, 1);
    if(d == 0.0) to_contract2<0, 1, 1, double>(contr, ta, tb).perform(true, tc);
    else to_contract2<0, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_i_pi_p(size_t ni, size_t np, double d) {

    // c_i = \sum_p a_{pi} b_p

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_i_pi_p(" << ni << ", " << np << ", "
        << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
    libtensor::index<1> ib1, ib2; ib2[0] = np - 1;
    libtensor::index<1> ic1, ic2; ic2[0] = ni - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<1> dimc(index_range<1>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<1, double, allocator> tc(dimc);
    dense_tensor<1, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<1, double> tcc(tc);
    dense_tensor_ctrl<1, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<1> ib; libtensor::index<1> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = p;
        ic[0] = i;
        abs_index<2> aa(ia, dima);
        abs_index<1> ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 0, 1> contr;
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<1, 0, 1, double>(contr, ta, tb).perform(true, tc);
    else to_contract2<1, 0, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_i_ip_p(size_t ni, size_t np, double d) {

    // c_i = \sum_p a_{ip} b_p

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_i_ip_p(" << ni << ", " << np << ", "
        << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
    libtensor::index<1> ib1, ib2; ib2[0] = np - 1;
    libtensor::index<1> ic1, ic2; ic2[0] = ni - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<1> dimc(index_range<1>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<1, double, allocator> tc(dimc);
    dense_tensor<1, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<1, double> tcc(tc);
    dense_tensor_ctrl<1, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<1> ib; libtensor::index<1> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p;
        ib[0] = p;
        ic[0] = i;
        abs_index<2> aa(ia, dima);
        abs_index<1> ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 0, 1> contr;
    contr.contract(1, 0);
    if(d == 0.0) to_contract2<1, 0, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 0, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_i_j(size_t ni, size_t nj, double d) {

    // c_{ij} = c_{ij} + d a_{i} b_{j}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_i_j(" << ni << ", " << nj
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<1> ia1, ia2; ia2[0] = ni - 1;
    libtensor::index<1> ib1, ib2; ib2[0] = nj - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<1> ia; libtensor::index<1> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        ia[0] = i;
        ib[0] = j;
        ic[0] = i; ic[1] = j;
        abs_index<1> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 0> contr;
    if(d == 0.0) to_contract2<1, 1, 0, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 0, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_j_i(size_t ni, size_t nj, double d) {

    // c_{ij} = c_{ij} + d a_{j} b_{i}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_j_i(" << ni << ", " << nj
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<1> ia1, ia2; ia2[0] = nj - 1;
    libtensor::index<1> ib1, ib2; ib2[0] = ni - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<1> ia; libtensor::index<1> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        ia[0] = j;
        ib[0] = i;
        ic[0] = i; ic[1] = j;
        abs_index<1> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 0> contr(permutation<2>().permute(0, 1));
    if(d == 0.0) to_contract2<1, 1, 0, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 0, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pi_pj(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{pi} b_{pj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pi_pj(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = p; ib[1] = j;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 1> contr;
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pi_jp(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{pi} b_{jp}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pi_jp(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = nj - 1; ib2[1] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = j; ib[1] = p;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 1> contr;
    contr.contract(0, 1);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_ip_pj(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{ip} b_{pj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_ip_pj(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p;
        ib[0] = p; ib[1] = j;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 1> contr;
    contr.contract(1, 0);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_ip_jp(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{ip} b_{jp}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_ip_jp(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = nj - 1; ib2[1] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p;
        ib[0] = j; ib[1] = p;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 1> contr;
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pj_pi(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{pj} b_{pi}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pj_pi(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = j;
        ib[0] = p; ib[1] = i;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc; permc.permute(0, 1);
    contraction2<1, 1, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pj_ip(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{pj} b_{ip}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pj_ip(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = j;
        ib[0] = i; ib[1] = p;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc; permc.permute(0, 1);
    contraction2<1, 1, 1> contr(permc);
    contr.contract(0, 1);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_jp_ip(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{jp} b_{ip}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_jp_ip(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = nj - 1; ia2[1] = np - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = j; ia[1] = p;
        ib[0] = i; ib[1] = p;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc; permc.permute(0, 1);
    contraction2<1, 1, 1> contr(permc);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_jp_pi(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{jp} b_{pi}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_jp_pi(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = nj - 1; ia2[1] = np - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = j; ia[1] = p;
        ib[0] = p; ib[1] = i;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc; permc.permute(0, 1);
    contraction2<1, 1, 1> contr(permc);
    contr.contract(1, 0);
    if(d == 0.0) to_contract2<1, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_p_pji(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{p} b_{pji}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_p_pji(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<1> ia1, ia2; ia2[0] = np - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1; ib2[2] = ni - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<1, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<1> ia; libtensor::index<3> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p;
        ib[0] = p; ib[1] = j; ib[2] = i;
        ic[0] = i; ic[1] = j;
        abs_index<1> aa(ia, dima);
        abs_index<3> ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc; permc.permute(0, 1);
    contraction2<0, 2, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<0, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<0, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pji_p(size_t ni, size_t nj, size_t np, double d) {

    // c_{ij} = \sum_p a_{pji} b_{p}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pji_p(" << ni << ", " << nj
        << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1; ia2[2] = ni - 1;
    libtensor::index<1> ib1, ib2; ib2[0] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<1, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<1> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = j; ia[2] = i;
        ib[0] = p;
        ic[0] = i; ic[1] = j;
        abs_index<3> aa(ia, dima);
        abs_index<1> ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc; permc.permute(0, 1);
    contraction2<2, 0, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<2, 0, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 0, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pi_pj_qi_jq(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    //  c_{ij} = \sum_p a^1_{pi} b^1_{pj} + \sum_q a^2_{qi} b^2_{jq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pi_pj_qi_jq(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2;
    ia1[0] = np - 1; ia1[1] = ni - 1;
    ia2[0] = nq - 1; ia2[1] = ni - 1;
    libtensor::index<2> ib1, ib2;
    ib1[0] = np - 1; ib1[1] = nj - 1;
    ib2[0] = nj - 1; ib2[1] = nq - 1;
    libtensor::index<2> ic1;
    ic1[0] = ni - 1; ic1[1] = nj - 1;
    dimensions<2> dima1(index_range<2>(libtensor::index<2>(), ia1));
    dimensions<2> dima2(index_range<2>(libtensor::index<2>(), ia2));
    dimensions<2> dimb1(index_range<2>(libtensor::index<2>(), ib1));
    dimensions<2> dimb2(index_range<2>(libtensor::index<2>(), ib2));
    dimensions<2> dimc(index_range<2>(libtensor::index<2>(), ic1));
    size_t sza1 = dima1.get_size(), sza2 = dima2.get_size(),
        szb1 = dimb1.get_size(), szb2 = dimb2.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta1(dima1), ta2(dima2);
    dense_tensor<2, double, allocator> tb1(dimb1), tb2(dimb2);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);
    double d1, d2;

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca1(ta1), tca2(ta2);
    dense_tensor_ctrl<2, double> tcb1(tb1), tcb2(tb2);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta1 = tca1.req_dataptr();
    double *dta2 = tca2.req_dataptr();
    double *dtb1 = tcb1.req_dataptr();
    double *dtb2 = tcb2.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //  Fill in random input

    for(size_t i = 0; i < sza1; i++) dta1[i] = drand48();
    for(size_t i = 0; i < sza2; i++) dta2[i] = drand48();
    for(size_t i = 0; i < szb1; i++) dtb1[i] = drand48();
    for(size_t i = 0; i < szb2; i++) dtb2[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];
    d1 = drand48();
    d2 = drand48();

    //  Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double k1 = (d == 0.0) ? d1 : d * d1;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = p; ib[1] = j;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima1), ab(ib, dimb1), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k1 *
            dta1[aa.get_abs_index()] * dtb1[ab.get_abs_index()];
    }
    }
    }
    double k2 = (d == 0.0) ? d2 : d * d2;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = q; ia[1] = i;
        ib[0] = j; ib[1] = q;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima2), ab(ib, dimb2), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k2 *
            dta2[aa.get_abs_index()] * dtb2[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++) {
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;
    }

    tca1.ret_dataptr(dta1); dta1 = 0; ta1.set_immutable();
    tca2.ret_dataptr(dta2); dta2 = 0; ta2.set_immutable();
    tcb1.ret_dataptr(dtb1); dtb1 = 0; tb1.set_immutable();
    tcb2.ret_dataptr(dtb2); dtb2 = 0; tb2.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //  Invoke the contraction routine

    contraction2<1, 1, 1> contr1, contr2;
    contr1.contract(0, 0);
    contr2.contract(0, 1);
    bool zero;
    double k;
    if(d == 0.0) {
        zero = true; k = 1.0;
    } else {
        zero = false; k = d;
    }
    to_contract2<1, 1, 1, double> op(contr1, ta1, tb1, d1 * k);
    op.add_args(contr2, ta2, tb2, d2 * k);
    op.perform(zero, tc);

    //  Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pi_pj_qi_qj(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    //  c_{ij} = \sum_p a^1_{pi} b^1_{pj} + \sum_q a^2_{qi} b^2_{qj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pi_pj_qi_qj(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2;
    ia1[0] = np - 1; ia1[1] = ni - 1;
    ia2[0] = nq - 1; ia2[1] = ni - 1;
    libtensor::index<2> ib1, ib2;
    ib1[0] = np - 1; ib1[1] = nj - 1;
    ib2[0] = nq - 1; ib2[1] = nj - 1;
    libtensor::index<2> ic1;
    ic1[0] = ni - 1; ic1[1] = nj - 1;
    dimensions<2> dima1(index_range<2>(libtensor::index<2>(), ia1));
    dimensions<2> dima2(index_range<2>(libtensor::index<2>(), ia2));
    dimensions<2> dimb1(index_range<2>(libtensor::index<2>(), ib1));
    dimensions<2> dimb2(index_range<2>(libtensor::index<2>(), ib2));
    dimensions<2> dimc(index_range<2>(libtensor::index<2>(), ic1));
    size_t sza1 = dima1.get_size(), sza2 = dima2.get_size(),
        szb1 = dimb1.get_size(), szb2 = dimb2.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta1(dima1), ta2(dima2);
    dense_tensor<2, double, allocator> tb1(dimb1), tb2(dimb2);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);
    double d1, d2;

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca1(ta1), tca2(ta2);
    dense_tensor_ctrl<2, double> tcb1(tb1), tcb2(tb2);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta1 = tca1.req_dataptr();
    double *dta2 = tca2.req_dataptr();
    double *dtb1 = tcb1.req_dataptr();
    double *dtb2 = tcb2.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //  Fill in random input

    for(size_t i = 0; i < sza1; i++) dta1[i] = drand48();
    for(size_t i = 0; i < sza2; i++) dta2[i] = drand48();
    for(size_t i = 0; i < szb1; i++) dtb1[i] = drand48();
    for(size_t i = 0; i < szb2; i++) dtb2[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];
    d1 = drand48();
    d2 = drand48();

    //  Generate reference data

    libtensor::index<2> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double k1 = (d == 0.0) ? d1 : d * d1;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = p; ib[1] = j;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima1), ab(ib, dimb1), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k1 *
            dta1[aa.get_abs_index()] * dtb1[ab.get_abs_index()];
    }
    }
    }
    double k2 = (d == 0.0) ? d2 : d * d2;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = q; ia[1] = i;
        ib[0] = q; ib[1] = j;
        ic[0] = i; ic[1] = j;
        abs_index<2> aa(ia, dima2), ab(ib, dimb2), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k2 *
            dta2[aa.get_abs_index()] * dtb2[ab.get_abs_index()];
    }
    }
    }
    for(size_t i = 0; i < szc; i++) {
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;
    }

    tca1.ret_dataptr(dta1); dta1 = 0; ta1.set_immutable();
    tca2.ret_dataptr(dta2); dta2 = 0; ta2.set_immutable();
    tcb1.ret_dataptr(dtb1); dtb1 = 0; tb1.set_immutable();
    tcb2.ret_dataptr(dtb2); dtb2 = 0; tb2.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //  Invoke the contraction routine

    contraction2<1, 1, 1> contr1, contr2;
    contr1.contract(0, 0);
    contr2.contract(0, 0);
    bool zero;
    double k;
    if(d == 0.0) {
        zero = true; k = 1.0;
    } else {
        zero = false; k = d;
    }
    to_contract2<1, 1, 1, double> op(contr1, ta1, tb1, d1 * k);
    op.add_args(contr2, ta2, tb2, d2 * k);
    op.perform(zero, tc);

    //  Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_ip_pkj(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = \sum_p a_{ip} b_{pkj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_ip_pkj(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nj - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p;
        ib[0] = p; ib[1] = k; ib[2] = j;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<2> aa(ia, dima);
        abs_index<3> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(1, 2); // ikj -> ijk
    contraction2<1, 2, 1> contr(permc);
    contr.contract(1, 0);
    if(d == 0.0) to_contract2<1, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pi_pkj(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = c_{ijk} + d \sum_p a_{pi} b_{pkj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pi_pkj(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nj - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = p; ib[1] = k; ib[2] = j;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<2> aa(ia, dima);
        abs_index<3> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(1, 2); // ikj -> ijk
    contraction2<1, 2, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<1, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pik_pj(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = c_{ijk} + d \sum_p a_{pik} b_{pj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pik_pj(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nk - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<2> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i; ia[2] = k;
        ib[0] = p; ib[1] = j;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<3> aa(ia, dima);
        abs_index<2> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(1, 2); // ikj -> ijk
    contraction2<2, 1, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<2, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pj_ipk(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = c_{ijk} + d \sum_p a_{pj} b_{ipk}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pj_ipk(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1; ib2[2] = nk - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = j;
        ib[0] = i; ib[1] = p; ib[2] = k;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<2> aa(ia, dima);
        abs_index<3> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(0, 1); // jik -> ijk
    contraction2<1, 2, 1> contr(permc);
    contr.contract(0, 1);
    if(d == 0.0) to_contract2<1, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pj_pik(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = c_{ijk} + d \sum_p a_{pj} b_{pik}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pj_pik(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1; ib2[2] = nk - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = j;
        ib[0] = p; ib[1] = i; ib[2] = k;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<2> aa(ia, dima);
        abs_index<3> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(0, 1); // jik -> ijk
    contraction2<1, 2, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<1, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pkj_ip(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = \sum_p a_{pkj} b_{ip}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pkj_ip(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = np - 1; ia2[1] = nk - 1; ia2[2] = nj - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<2> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = k; ia[2] = j;
        ib[0] = i; ib[1] = p;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<3> aa(ia, dima);
        abs_index<2> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(0, 2); // kji -> ijk
    contraction2<2, 1, 1> contr(permc);
    contr.contract(0, 1);
    if(d == 0.0) to_contract2<2, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pkj_pi(size_t ni, size_t nj, size_t nk, size_t np, double d) {

    // c_{ijk} = \sum_p a_{pkj} b_{pi}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pkj_pi(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = np - 1; ia2[1] = nk - 1; ia2[2] = nj - 1;
    libtensor::index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
    libtensor::index<3> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<2> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = k; ia[2] = j;
        ib[0] = p; ib[1] = i;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<3> aa(ia, dima);
        abs_index<2> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(0, 2); // kji -> ijk
    contraction2<2, 1, 1> contr(permc);
    contr.contract(0, 0);
    if(d == 0.0) to_contract2<2, 1, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 1, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_pqi_pjq(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    // c_{ij} = \sum_{pq} a_{pqi} b_{pjq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pqi_pjq(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = np - 1; ia2[1] = nq - 1; ia2[2] = ni - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1; ib2[2] = nq - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = q; ia[2] = i;
        ib[0] = p; ib[1] = j; ib[2] = q;
        ic[0] = i; ic[1] = j;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 2> contr;
    contr.contract(0, 0);
    contr.contract(1, 2);
    if(d == 0.0) to_contract2<1, 1, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}

int test_ij_ipq_jqp(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    // c_{ij} = \sum_{pq} a_{ipq} b_{jqp}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_ipq_jqp(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = nj - 1; ib2[1] = nq - 1; ib2[2] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = i; ia[1] = p; ia[2] = q;
        ib[0] = j; ib[1] = q; ib[2] = p;
        ic[0] = i; ic[1] = j;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 2> contr;
    contr.contract(1, 2);
    contr.contract(2, 1);
    if(d == 0.0) to_contract2<1, 1, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}

int test_ij_jpq_iqp(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    // c_{ij} = \sum_{pq} a_{jpq} b_{iqp}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_jpq_iqp(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2; ia2[0] = nj - 1; ia2[1] = np - 1; ia2[2] = nq - 1;
    libtensor::index<3> ib1, ib2; ib2[0] = ni - 1; ib2[1] = nq - 1; ib2[2] = np - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = j; ia[1] = p; ia[2] = q;
        ib[0] = i; ib[1] = q; ib[2] = p;
        ic[0] = i; ic[1] = j;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 2> contr(permutation<2>().permute(0, 1));
    contr.contract(1, 2);
    contr.contract(2, 1);
    if(d == 0.0) to_contract2<1, 1, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}

int test_ij_jipq_qp(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    // c_{ij} = \sum_{pq} a_{jipq} b_{qp}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_jipq_qp(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = nj - 1; ia2[1] = ni - 1; ia2[2] = np - 1; ia2[3] = nq - 1;
    libtensor::index<2> ib1, ib2;
    ib2[0] = nq - 1; ib2[1] = np - 1;
    libtensor::index<2> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<2> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = j; ia[1] = i; ia[2] = p; ia[3] = q;
        ib[0] = q; ib[1] = p;
        ic[0] = i; ic[1] = j;
        abs_index<4> aa(ia, dima);
        abs_index<2> ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 0, 2> contr(permutation<2>().permute(0, 1));
    contr.contract(2, 1);
    contr.contract(3, 0);
    if(d == 0.0) to_contract2<2, 0, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 0, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}

int test_ij_pq_ijpq(size_t ni, size_t nj, size_t np, size_t nq) { 

    // c_{ij} = \sum_{pq} a_{pq} b_{ijpq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pq_ijpq(" << ni << ", " << nj
        << ", " << np << ", " << nq << ")";
    std::string tns = tnss.str();

    libtensor::index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1;
    libtensor::index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=np-1; ib2[3]=nq-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<4> ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        ia[0]=p; ia[1]=q;
        ib[0]=i; ib[1]=j; ib[2]=p; ib[3]=q;
            abs_index<2> aa(ia, dima);
            abs_index<4> ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        dtc2[ac.get_abs_index()] = cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc;
    contraction2<0, 2, 2> contr(permc);
    contr.contract(0, 2);
    contr.contract(1, 3);

    to_contract2<0, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}

int test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np, size_t nq, double d) {

    // c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pq_ijpq_a(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    libtensor::index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1;
    libtensor::index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=np-1; ib2[3]=nq-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<4> ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        ia[0]=p; ia[1]=q;
        ib[0]=i; ib[1]=j; ib[2]=p; ib[3]=q;
        abs_index<2> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        dtc2[ac.get_abs_index()] += d*cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc;
    contraction2<0, 2, 2> contr(permc);
    contr.contract(0, 2);
    contr.contract(1, 3);

    to_contract2<0, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ijk_kjpq_iqp(size_t ni, size_t nj, size_t nk,
    size_t np, size_t nq, double d) {

    // c_{ijk} = \sum_{pq} a_{kjpq} b_{iqp}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_kjpq_iqp(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = nk - 1; ia2[1] = nj - 1; ia2[2] = np - 1; ia2[3] = nq - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = nq - 1; ib2[2] = np - 1;
    libtensor::index<3> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = k; ia[1] = j; ia[2] = p; ia[3] = q;
        ib[0] = i; ib[1] = q; ib[2] = p;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<4> aa(ia, dima);
        abs_index<3> ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 1, 2> contr(permutation<3>().permute(0, 2));
    contr.contract(2, 2);
    contr.contract(3, 1);
    if(d == 0.0) to_contract2<2, 1, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 1, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pkiq_pjq(size_t ni, size_t nj, size_t nk,
    size_t np, size_t nq, double d) {

    // c_{ijk} = c_{ijk} + d \sum_{pq} a_{pkiq} b_{pjq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pkiq_pjq(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = nk - 1; ia2[2] = ni - 1; ia2[3] = nq - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nj - 1; ib2[2] = nq - 1;
    libtensor::index<3> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = k; ia[2] = i; ia[3] = q;
        ib[0] = p; ib[1] = j; ib[2] = q;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<4> aa(ia, dima);
        abs_index<3> ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<3> permc;
    permc.permute(0, 1).permute(1, 2); // kij -> ijk
    contraction2<2, 1, 2> contr(permc);
    contr.contract(0, 0);
    contr.contract(3, 2);
    if(d == 0.0) to_contract2<2, 1, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 1, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pqj_iqpk(size_t ni, size_t nj, size_t nk,
    size_t np, size_t nq, double d) {

    // c_{ijk} = \sum_{pq} a_{pqj} b_{iqpk}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pqj_iqpk(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = nq - 1; ia2[2] = nj - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = nq - 1; ib2[2] = np - 1; ib2[3] = nk - 1;
    libtensor::index<3> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<4> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = q; ia[2] = j;
        ib[0] = i; ib[1] = q; ib[2] = p; ib[3] = k;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<3> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        abs_index<3> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 2, 2> contr(permutation<3>().permute(0, 1));
    contr.contract(0, 2);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<1, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijk_pqji_qpk(size_t ni, size_t nj, size_t nk,
    size_t np, size_t nq, double d) {

    // c_{ijk} = \sum_{pq} a_{pqji} b_{qpk}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijk_pqji_qpk(" << ni << ", " << nj
        << ", " << nk << ", " << np << ", " << nq << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = nq - 1; ia2[2] = nj - 1; ia2[3] = ni - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = nq - 1; ib2[1] = np - 1; ib2[2] = nk - 1;
    libtensor::index<3> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<3, double> tcc(tc);
    dense_tensor_ctrl<3, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<3> ib; libtensor::index<3> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = q; ia[2] = j; ia[3] = i;
        ib[0] = q; ib[1] = p; ib[2] = k;
        ic[0] = i; ic[1] = j; ic[2] = k;
        abs_index<4> aa(ia, dima);
        abs_index<3> ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 1, 2> contr(permutation<3>().permute(0, 1));
    contr.contract(0, 1);
    contr.contract(1, 0);
    if(d == 0.0) to_contract2<2, 1, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 1, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_ikp_jpl(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ikp} b_{jpl}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ikp_jpl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nk - 1; ia2[2] = np - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = np - 1; ib2[2] = nl - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = k; ia[2] = p;
        ib[0] = j; ib[1] = p; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    permc.permute(1, 2); // ikjl -> ijkl
    contraction2<2, 2, 1> contr(permc);
    contr.contract(2, 1);
    if(d == 0.0) to_contract2<2, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_ipk_jpl(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ikp} b_{jpl}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ipk_jpl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nk - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = np - 1; ib2[2] = nl - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p; ia[2] = k;
        ib[0] = j; ib[1] = p; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    permc.permute(1, 2); // ikjl -> ijkl
    contraction2<2, 2, 1> contr(permc);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<2, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_ipl_jpk(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ipl} b_{jpk}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ipl_jpk(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = np - 1; ib2[2] = nk - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p; ia[2] = l;
        ib[0] = j; ib[1] = p; ib[2] = k;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc; permc.permute(1, 2).permute(2, 3); // iljk->ijkl
    contraction2<2, 2, 1> contr(permc);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<2, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_jkp_ipl(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{jkp} b_{ipl}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_jkp_ipl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = nj - 1; ia2[1] = nk - 1; ia2[2] = np - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = np - 1; ib2[2] = nl - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = j; ia[1] = k; ia[2] = p;
        ib[0] = i; ib[1] = p; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    permc.permute(0, 2).permute(1, 2); // jkil -> ijkl
    contraction2<2, 2, 1> contr(permc);
    contr.contract(2, 1);
    if(d == 0.0) to_contract2<2, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_jpl_ipk(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{jpl} b_{ipk}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_jpl_ipk(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = nj - 1; ia2[1] = np - 1; ia2[2] = nl - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = np - 1; ib2[2] = nk - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = j; ia[1] = p; ia[2] = l;
        ib[0] = i; ib[1] = p; ib[2] = k;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    // jlik -> iljk -> ijlk -> ijkl
    permc.permute(0, 2).permute(1, 2).permute(2, 3);
    contraction2<2, 2, 1> contr(permc);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<2, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np, size_t nq, size_t nr, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a^1_{jpl} b^1_{ipk}
    //                     + d \sum_{q} a^2_{jiq} b^2_{kql}
    //                     + d \sum_{r} a^3_{jlr} b^3_{ikr}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_jpl_ipk_jiq_kql_jlr_ikr("
        << ni << ", " << nj << ", " << nk << ", " << nl << ", " << np << ", "
        << nq << ", " << nr << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2, ia3;
    ia1[0] = nj - 1; ia1[1] = np - 1; ia1[2] = nl - 1;
    ia2[0] = nj - 1; ia2[1] = ni - 1; ia2[2] = nq - 1;
    ia3[0] = nj - 1; ia3[1] = nl - 1; ia3[2] = nr - 1;
    libtensor::index<3> ib1, ib2, ib3;
    ib1[0] = ni - 1; ib1[1] = np - 1; ib1[2] = nk - 1;
    ib2[0] = nk - 1; ib2[1] = nq - 1; ib2[2] = nl - 1;
    ib3[0] = ni - 1; ib3[1] = nk - 1; ib3[2] = nr - 1;
    libtensor::index<4> ic1;
    ic1[0] = ni - 1; ic1[1] = nj - 1; ic1[2] = nk - 1; ic1[3] = nl - 1;
    dimensions<3> dima1(index_range<3>(libtensor::index<3>(), ia1));
    dimensions<3> dima2(index_range<3>(libtensor::index<3>(), ia2));
    dimensions<3> dima3(index_range<3>(libtensor::index<3>(), ia3));
    dimensions<3> dimb1(index_range<3>(libtensor::index<3>(), ib1));
    dimensions<3> dimb2(index_range<3>(libtensor::index<3>(), ib2));
    dimensions<3> dimb3(index_range<3>(libtensor::index<3>(), ib3));
    dimensions<4> dimc(index_range<4>(libtensor::index<4>(), ic1));
    size_t sza1 = dima1.get_size(), sza2 = dima2.get_size(),
        sza3 = dima3.get_size(), szb1 = dimb1.get_size(),
        szb2 = dimb2.get_size(), szb3 = dimb3.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta1(dima1), ta2(dima2), ta3(dima3);
    dense_tensor<3, double, allocator> tb1(dimb1), tb2(dimb2), tb3(dimb3);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);
    double d1, d2, d3;

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca1(ta1), tca2(ta2), tca3(ta3);
    dense_tensor_ctrl<3, double> tcb1(tb1), tcb2(tb2), tcb3(tb3);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta1 = tca1.req_dataptr();
    double *dta2 = tca2.req_dataptr();
    double *dta3 = tca3.req_dataptr();
    double *dtb1 = tcb1.req_dataptr();
    double *dtb2 = tcb2.req_dataptr();
    double *dtb3 = tcb3.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //  Fill in random input

    for(size_t i = 0; i < sza1; i++) dta1[i] = drand48();
    for(size_t i = 0; i < sza2; i++) dta2[i] = drand48();
    for(size_t i = 0; i < sza3; i++) dta3[i] = drand48();
    for(size_t i = 0; i < szb1; i++) dtb1[i] = drand48();
    for(size_t i = 0; i < szb2; i++) dtb2[i] = drand48();
    for(size_t i = 0; i < szb3; i++) dtb3[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];
    d1 = drand48() - 0.5;
    d2 = drand48() - 0.5;
    d3 = drand48() - 0.5;

    //  Generate reference data

    libtensor::index<3> ia; libtensor::index<3> ib; libtensor::index<4> ic;
    double k1 = (d == 0.0) ? d1 : d * d1;
    // \sum_{p} a^1_{jpl} b^1_{ipk}
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = j; ia[1] = p; ia[2] = l;
        ib[0] = i; ib[1] = p; ib[2] = k;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima1), ab(ib, dimb1);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k1 *
            dta1[aa.get_abs_index()] * dtb1[ab.get_abs_index()];
    }
    }
    }
    }
    }
    double k2 = (d == 0.0) ? d2 : d * d2;
    // \sum_{q} a^2_{jiq} b^2_{kql}
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = j; ia[1] = i; ia[2] = q;
        ib[0] = k; ib[1] = q; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima2), ab(ib, dimb2);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k2 *
            dta2[aa.get_abs_index()] * dtb2[ab.get_abs_index()];
    }
    }
    }
    }
    }
    double k3 = (d == 0.0) ? d3 : d * d3;
    // \sum_{r} a^3_{jlr} b^3_{ikr}
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t r = 0; r < nr; r++) {
        ia[0] = j; ia[1] = l; ia[2] = r;
        ib[0] = i; ib[1] = k; ib[2] = r;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<3> aa(ia, dima3), ab(ib, dimb3);
        abs_index<4> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += k3 *
            dta3[aa.get_abs_index()] * dtb3[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++) {
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;
    }

    tca1.ret_dataptr(dta1); dta1 = 0; ta1.set_immutable();
    tca2.ret_dataptr(dta2); dta2 = 0; ta2.set_immutable();
    tca3.ret_dataptr(dta3); dta3 = 0; ta3.set_immutable();
    tcb1.ret_dataptr(dtb1); dtb1 = 0; tb1.set_immutable();
    tcb2.ret_dataptr(dtb2); dtb2 = 0; tb2.set_immutable();
    tcb3.ret_dataptr(dtb3); dtb3 = 0; tb3.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    //  Invoke the contraction routine

    permutation<4> permc1, permc2, permc3;
    // jlik -> iljk -> ijlk -> ijkl
    permc1.permute(0, 2).permute(1, 2).permute(2, 3);
    // jikl -> ijkl
    permc2.permute(0, 1);
    // jlik -> iljk -> ijlk -> ijkl
    permc3.permute(0, 2).permute(1, 2).permute(2, 3);
    contraction2<2, 2, 1> contr1(permc1), contr2(permc2), contr3(permc3);
    contr1.contract(1, 1);
    contr2.contract(2, 1);
    contr3.contract(2, 2);
    bool zero;
    double k;
    if(d == 0.0) {
        zero = true;
        k = 1.0;
    } else {
        zero = false;
        k = d;
    }
    to_contract2<2, 2, 1, double> op(contr1, ta1, tb1, d1 * k);
    op.add_args(contr2, ta2, tb2, d2 * k);
    op.add_args(contr3, ta3, tb3, d3 * k);
    op.perform(zero, tc);

    //  Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijklm_ikp_jpml(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t nm, size_t np, double d) {

    // c_{ijklm} = c_{ijklm} + d \sum_{p} a_{ikp} b_{jpml}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijklm_ikp_jpml(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << nm << ", " << np
        << ", "  << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nk - 1; ia2[2] = np - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = np - 1; ib2[2] = nm - 1; ib2[3] = nl - 1;
    libtensor::index<5> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    ic2[4] = nm - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<5> dimc(index_range<5>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<5, double, allocator> tc(dimc);
    dense_tensor<5, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<5, double> tcc(tc);
    dense_tensor_ctrl<5, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<4> ib; libtensor::index<5> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t m = 0; m < nm; m++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = k; ia[2] = p;
        ib[0] = j; ib[1] = p; ib[2] = m; ib[3] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l; ic[4] = m;
        abs_index<3> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        abs_index<5> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<5> permc;
    // ikjml -> ijkml -> ijklm
    permc.permute(1, 2).permute(3, 4);
    contraction2<2, 3, 1> contr(permc);
    contr.contract(2, 1);
    if(d == 0.0) to_contract2<2, 3, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 3, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<5>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijklm_ipkm_jpl(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t nm, size_t np, double d) {

    // c_{ijklm} = \sum_{p} a_{ipkm} b_{jpl}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijklm_ipkm_jpl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << nm << ", " << np
        << ", "  << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nk - 1; ia2[3] = nm - 1;
    libtensor::index<3> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = np - 1; ib2[2] = nl - 1;
    libtensor::index<5> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    ic2[4] = nm - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<3> dimb(index_range<3>(ib1, ib2));
    dimensions<5> dimc(index_range<5>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<5, double, allocator> tc(dimc);
    dense_tensor<5, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<5, double> tcc(tc);
    dense_tensor_ctrl<5, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<3> ib; libtensor::index<5> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t m = 0; m < nm; m++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = i; ia[1] = p; ia[2] = k; ia[3] = m;
        ib[0] = j; ib[1] = p; ib[2] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l; ic[4] = m;
        abs_index<4> aa(ia, dima);
        abs_index<3> ab(ib, dimb);
        abs_index<5> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<5> permc;
    // ikmjl -> ijmkl -> ijkml -> ijklm
    permc.permute(1, 3).permute(2, 3).permute(3, 4);
    contraction2<3, 2, 1> contr(permc);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<3, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<3, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<5>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijklm_jlp_ipkm(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t nm, size_t np, double d) {

    // c_{ijklm} = \sum_{p} a_{jlp} b_{ipkm}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijklm_jlp_ipkm(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << nm << ", " << np
        << ", "  << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<3> ia1, ia2;
    ia2[0] = nj - 1; ia2[1] = nl - 1; ia2[2] = np - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = np - 1; ib2[2] = nk - 1; ib2[3] = nm - 1;
    libtensor::index<5> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    ic2[4] = nm - 1;
    dimensions<3> dima(index_range<3>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<5> dimc(index_range<5>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<5, double, allocator> tc(dimc);
    dense_tensor<5, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<5, double> tcc(tc);
    dense_tensor_ctrl<5, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<3> ia; libtensor::index<4> ib; libtensor::index<5> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t m = 0; m < nm; m++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = j; ia[1] = l; ia[2] = p;
        ib[0] = i; ib[1] = p; ib[2] = k; ib[3] = m;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l; ic[4] = m;
        abs_index<3> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        abs_index<5> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<5> permc;
    // jlikm -> iljkm -> ijlkm -> ijklm
    permc.permute(0, 2).permute(1, 2).permute(2, 3);
    contraction2<2, 3, 1> contr(permc);
    contr.contract(2, 1);
    if(d == 0.0) to_contract2<2, 3, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 3, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<5>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijklmn_kjmp_ipln(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t nm, size_t nn, size_t np, double d) {

    // c_{ijklmn} = c_{ijklmn} + d \sum_{p} a_{kjmp} b_{ipln}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijklmn_kjmp_ipln(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << nm << ", " << nn
        << ", "  << np << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = nk - 1; ia2[1] = nj - 1; ia2[2] = nm - 1; ia2[3] = np - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = np - 1; ib2[2] = nl - 1; ib2[3] = nn - 1;
    libtensor::index<6> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    ic2[4] = nm - 1; ic2[5] = nn - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<6> dimc(index_range<6>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<6, double, allocator> tc(dimc);
    dense_tensor<6, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<6, double> tcc(tc);
    dense_tensor_ctrl<6, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<6> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t m = 0; m < nm; m++) {
    for(size_t n = 0; n < nn; n++) {
    for(size_t p = 0; p < np; p++) {
        ia[0] = k; ia[1] = j; ia[2] = m; ia[3] = p;
        ib[0] = i; ib[1] = p; ib[2] = l; ib[3] = n;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l; ic[4] = m;
        ic[5] = n;
        abs_index<4> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        abs_index<6> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<6> permc;
    // kjmiln -> ijmkln -> ijkmln -> ijklmn
    permc.permute(0, 3).permute(2, 3).permute(3, 4);
    contraction2<3, 3, 1> contr(permc);
    contr.contract(3, 1);
    if(d == 0.0) to_contract2<3, 3, 1, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<3, 3, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<6>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_iplq_kpjq(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{iplq} b_{kpjq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_iplq_kpjq(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = i; ia[1] = p; ia[2] = l; ia[3] = q;
        ib[0] = k; ib[1] = p; ib[2] = j; ib[3] = q;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(1, 1);
    contr.contract(3, 3);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_iplq_pkjq(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{iplq} b_{pkjq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_iplq_pkjq(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = i; ia[1] = p; ia[2] = l; ia[3] = q;
        ib[0] = p; ib[1] = k; ib[2] = j; ib[3] = q;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(1, 0);
    contr.contract(3, 3);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_iplq_pkqj(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{iplq} b_{pkqj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_iplq_pkqj(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = i; ia[1] = p; ia[2] = l; ia[3] = q;
        ib[0] = p; ib[1] = k; ib[2] = q; ib[3] = j;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(1, 0);
    contr.contract(3, 2);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_ipql_kpqj(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{ipql} b_{kpqj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ipql_kpqj(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = i; ia[1] = p; ia[2] = q; ia[3] = l;
        ib[0] = k; ib[1] = p; ib[2] = q; ib[3] = j;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(1, 1);
    contr.contract(2, 2);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_ipql_pkqj(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{ipql} b_{pkqj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ipql_pkqj(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = i; ia[1] = p; ia[2] = q; ia[3] = l;
        ib[0] = p; ib[1] = k; ib[2] = q; ib[3] = j;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(1, 0);
    contr.contract(2, 2);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_pilq_kpjq(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) { 

    // c_{ijkl} = \sum_{pq} a_{pilq} b_{kpjq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_pilq_kpjq(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = i; ia[2] = l; ia[3] = q;
        ib[0] = k; ib[1] = p; ib[2] = j; ib[3] = q;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(0, 1);
    contr.contract(3, 3);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_pilq_pkjq(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{pilq} b_{pkjq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_pilq_pkjq(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = i; ia[2] = l; ia[3] = q;
        ib[0] = p; ib[1] = k; ib[2] = j; ib[3] = q;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(0, 0);
    contr.contract(3, 3);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_piql_kpqj(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{piql} b_{kpqj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_piql_kpqj(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = i; ia[2] = q; ia[3] = l;
        ib[0] = k; ib[1] = p; ib[2] = q; ib[3] = j;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(0, 1);
    contr.contract(2, 2);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_piql_pkqj(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = \sum_{pq} a_{piql} b_{pkqj}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_piql_pkqj(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = i; ia[2] = q; ia[3] = l;
        ib[0] = p; ib[1] = k; ib[2] = q; ib[3] = j;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
    contr.contract(0, 0);
    contr.contract(2, 2);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_pqkj_iqpl(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pqkj} b_{iqpl}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_pqkj_iqpl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = nq - 1; ia2[2] = nk - 1; ia2[3] = nj - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = ni - 1; ib2[1] = nq - 1; ib2[2] = np - 1; ib2[3] = nl - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = q; ia[2] = k; ia[3] = j;
        ib[0] = i; ib[1] = q; ib[2] = p; ib[3] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    permc.permute(0, 2);  // kjil -> ijkl
    contraction2<2, 2, 2> contr(permc);
    contr.contract(0, 2);
    contr.contract(1, 1);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_pqkj_qipl(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, size_t nq, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pqkj} b_{qipl}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_pqkj_qipl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << nq
        << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = nq - 1; ia2[2] = nk - 1; ia2[3] = nj - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = nq - 1; ib2[1] = ni - 1; ib2[2] = np - 1; ib2[3] = nl - 1;
    libtensor::index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<4> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
        ia[0] = p; ia[1] = q; ia[2] = k; ia[3] = j;
        ib[0] = q; ib[1] = i; ib[2] = p; ib[3] = l;
        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    permc.permute(0, 2);  // kjil -> ijkl
    contraction2<2, 2, 2> contr(permc);
    contr.contract(0, 2);
    contr.contract(1, 0);
    if(d == 0.0) to_contract2<2, 2, 2, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<2, 2, 2, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np,
    size_t nq, size_t nr) {

    // c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_ipqr_jpqr(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << nr << ")";
    std::string tns = tnss.str();

    libtensor::index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
    libtensor::index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

    // Generate reference data

    libtensor::index<4> ia, ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        for(size_t r=0; r<nr; r++) {
        ia[0]=i; ia[1]=p; ia[2]=q; ia[3]=r;
        ib[0]=j; ib[1]=p; ib[2]=q; ib[3]=r;
        abs_index<4> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        }
        dtc2[ac.get_abs_index()] = cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc;
    contraction2<1, 1, 3> contr(permc);
    contr.contract(1, 1);
    contr.contract(2, 2);
    contr.contract(3, 3);

    to_contract2<1, 1, 3, double>(contr, ta, tb, 1.0).perform(true, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np,
    size_t nq, size_t nr, double d) {

    // c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_ipqr_jpqr_a(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
    std::string tns = tnss.str();

    libtensor::index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
    libtensor::index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

    // Generate reference data

    libtensor::index<4> ia, ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        for(size_t r=0; r<nr; r++) {
        ia[0]=i; ia[1]=p; ia[2]=q; ia[3]=r;
        ib[0]=j; ib[1]=p; ib[2]=q; ib[3]=r;
        abs_index<4> aa(ia, dima), ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        }
        dtc2[ac.get_abs_index()] += d*cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc;
    contraction2<1, 1, 3> contr(permc);
    contr.contract(1, 1);
    contr.contract(2, 2);
    contr.contract(3, 3);

    to_contract2<1, 1, 3, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}

int test_ij_ipqr_pjrq(size_t ni, size_t nj, size_t np, size_t nq, size_t nr,
    double d) {

    // c_{ij} = \sum_{pq} a_{ipqr} b_{pjrq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_ipqr_pjrq(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
    std::string tns = tnss.str();

    try {

    libtensor::index<4> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1; ia2[3] = nr - 1;
    libtensor::index<4> ib1, ib2;
    ib2[0] = np - 1; ib2[1] = nj - 1; ib2[2] = nr - 1; ib2[3] = nq - 1;
    libtensor::index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<4> dima(index_range<4>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
    if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
    else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

    // Generate reference data

    libtensor::index<4> ia; libtensor::index<4> ib; libtensor::index<2> ic;
    double d1 = (d == 0.0) ? 1.0 : d;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t p = 0; p < np; p++) {
    for(size_t q = 0; q < nq; q++) {
    for(size_t r = 0; r < nr; r++) {
        ia[0] = i; ia[1] = p; ia[2] = q; ia[3] = r;
        ib[0] = p; ib[1] = j; ib[2] = r; ib[3] = q;
        ic[0] = i; ic[1] = j;
        abs_index<4> aa(ia, dima), ab(ib, dimb);
        abs_index<2> ac(ic, dimc);
        dtc2[ac.get_abs_index()] += d1 *
        dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
    }
    }
    }
    }
    }
    for(size_t i = 0; i < szc; i++)
        if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

    tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = 0;
    tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    contraction2<1, 1, 3> contr;
    contr.contract(1, 0);
    contr.contract(2, 3);
    contr.contract(3, 2);
    if(d == 0.0) to_contract2<1, 1, 3, double>(contr, ta, tb, 1.0).perform(true, tc);
    else to_contract2<1, 1, 3, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

    } catch(exception &e) {
        return fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}

int test_ij_jpqr_iprq(size_t ni, size_t nj, size_t np,
    size_t nq, size_t nr, double d) {

    // c_{ij} = c_{ij} + d \sum_{pqr} a_{jpqr} b_{iprq}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_jpqr_iprq(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
    std::string tns = tnss.str();

    libtensor::index<4> ia1, ia2; ia2[0]=nj-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
    libtensor::index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=np-1; ib2[2]=nr-1; ib2[3]=nq-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

    // Generate reference data

    libtensor::index<4> ia, ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        for(size_t r=0; r<nr; r++) {
        ia[0]=j; ia[1]=p; ia[2]=q; ia[3]=r;
        ib[0]=i; ib[1]=p; ib[2]=r; ib[3]=q;
        abs_index<4> aa(ia, dima), ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        }
        if(d == 0.0) dtc2[ac.get_abs_index()] = cij;
        else dtc2[ac.get_abs_index()] += d*cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    //~ contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
    contraction2<1, 1, 3> contr;
    contr.contract(1, 1);
    contr.contract(2, 3);
    contr.contract(3, 2);

    //~ to_contract2<1, 1, 3, double> op(contr, ta, tb);
    to_contract2<1, 1, 3, double> op(contr, tb, ta, (d != 0 ? d : 1.0));
    if(d == 0.0) op.perform(true, tc);
    else op.perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ij_pqir_pqjr(size_t ni, size_t nj,
    size_t np, size_t nq, size_t nr) {

    // c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pqir_pqjr(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << nr << ")";
    std::string tns = tnss.str();

    libtensor::index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
    libtensor::index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

    // Generate reference data

    libtensor::index<4> ia, ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        for(size_t r=0; r<nr; r++) {
        ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
        ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
        abs_index<4> aa(ia, dima), ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        }
        dtc2[ac.get_abs_index()] = cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc;
    contraction2<1, 1, 3> contr(permc);
    contr.contract(0, 0);
    contr.contract(1, 1);
    contr.contract(3, 3);

    to_contract2<1, 1, 3, double>(contr, ta, tb, 1.0).perform(true, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ij_pqir_pqjr_a(size_t ni, size_t nj, size_t np,
    size_t nq, size_t nr, double d) {

    // c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr}

    std::stringstream tnss;
    tnss << "tod_contract2_test::test_ij_pqir_pqjr_a(" << ni << ", " << nj
        << ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
    std::string tns = tnss.str();

    libtensor::index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
    libtensor::index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
    libtensor::index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
    index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<4, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<4, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<2, double> tcc(tc);
    dense_tensor_ctrl<2, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

    // Generate reference data

    libtensor::index<4> ia, ib; libtensor::index<2> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
        ic[0]=i; ic[1]=j;
        abs_index<2> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        for(size_t q=0; q<nq; q++) {
        for(size_t r=0; r<nr; r++) {
        ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
        ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
        abs_index<4> aa(ia, dima), ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        }
        }
        dtc2[ac.get_abs_index()] += d*cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<2> permc;
    contraction2<1, 1, 3> contr(permc);
    contr.contract(0, 0);
    contr.contract(1, 1);
    contr.contract(3, 3);

    to_contract2<1, 1, 3, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ijkl_pi_jklp(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np) {

    //
    // c_{ijkl} = \sum_p a_{pi} b_{jklp}
    //

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_pi_jklp(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ")";

    try {

    libtensor::index<2> ia1, ia2;
    libtensor::index<4> ib1, ib2;
    libtensor::index<4> ic1, ic2;
    ia2[0] = np - 1; ia2[1] = ni - 1;
    ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = nl - 1; ib2[3] = np - 1;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cijkl_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //
    // Fill in random input
    //

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();

    //
    // Generate reference data
    //

    libtensor::index<2> ia; libtensor::index<4> ib, ic;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {

        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aic(ic, dimc);
        double cijkl = 0.0;
        for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = j; ib[1] = k; ib[2] = l; ib[3] = p;
        abs_index<2> aia(ia, dima);
        abs_index<4> aib(ib, dimb);
        cijkl += dta[aia.get_abs_index()]*
         dtb[aib.get_abs_index()];
        }
        dtc2[aic.get_abs_index()] = cijkl;
        if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    //
    // Invoke the contraction routine
    //

    contraction2<1, 3, 1> contr;
    contr.contract(0, 3);

    to_contract2<1, 3, 1, double>(contr, ta, tb, 1.0).perform(true, tc);

    //
    // Compare against the reference
    //

    compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
        cijkl_max*k_thresh);

    } catch(exception &e) {
        return fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_pi_jklp_a(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    //
    // c_{ijkl} = c_{ijkl} + d * \sum_p a_{pi} b_{jklp}
    //

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_pi_jklp_a(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";

    try {

    libtensor::index<2> ia1, ia2;
    libtensor::index<4> ib1, ib2;
    libtensor::index<4> ic1, ic2;
    ia2[0] = np - 1; ia2[1] = ni - 1;
    ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = nl - 1; ib2[3] = np - 1;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<4> dimb(index_range<4>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cijkl_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //
    // Fill in random input
    //

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = dtc2[i] = drand48();

    //
    // Generate reference data
    //

    libtensor::index<2> ia; libtensor::index<4> ib, ic;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {

        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        abs_index<4> aic(ic, dimc);
        double cijkl = 0.0;
        for(size_t p = 0; p < np; p++) {
        ia[0] = p; ia[1] = i;
        ib[0] = j; ib[1] = k; ib[2] = l; ib[3] = p;
        abs_index<2> aia(ia, dima);
        abs_index<4> aib(ib, dimb);
        cijkl += dta[aia.get_abs_index()]*
         dtb[aib.get_abs_index()];
        }
        dtc2[aic.get_abs_index()] += d*cijkl;
        if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    //
    // Invoke the contraction routine
    //

    contraction2<1, 3, 1> contr;
    contr.contract(0, 3);

    to_contract2<1, 3, 1, double>(contr, ta, tb, d).perform(false, tc);

    //
    // Compare against the reference
    //

    compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
        cijkl_max*k_thresh);

    } catch(exception &e) {
        return fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_jikl_pi_jpkl(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np) {

    // c_{jikl} = \sum_p a_{pi} b_{jpkl}

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_jikl_pi_jpkl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ")";
    std::string tns = tnss.str();

    libtensor::index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
    libtensor::index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
    libtensor::index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

    index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<4> ib, ic;
    for(size_t j=0; j<nj; j++) {
    for(size_t i=0; i<ni; i++) {
    for(size_t k=0; k<nk; k++) {
    for(size_t l=0; l<nl; l++) {
        ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
        abs_index<4> ac(ic, dimc);
        double cjikl = 0.0;
        for(size_t p=0; p<np; p++) {
        ia[0]=p; ia[1]=i;
        ib[0]=j; ib[1]=p; ib[2]=k; ib[3]=l;
        abs_index<2> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        cjikl += dta[aa.get_abs_index()]*
         dtb[ab.get_abs_index()];
        }
        dtc2[ac.get_abs_index()] = cjikl;
        if(fabs(cjikl) > cij_max) cij_max = fabs(cjikl);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc; permc.permute(0, 1);
    contraction2<1, 3, 1> contr(permc);
    contr.contract(0, 1);

    to_contract2<1, 3, 1, double>(contr, ta, tb, 1.0).perform(true, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_jikl_pi_jpkl_a(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{jikl} = c_{jikl} + d \sum_p a_{pi} b_{jpkl}

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_jikl_pi_jpkl_a(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    libtensor::index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
    libtensor::index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
    libtensor::index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

    index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
    index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
    index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<4, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

    // Generate reference data

    libtensor::index<2> ia; libtensor::index<4> ib, ic;
    for(size_t j=0; j<nj; j++) {
    for(size_t i=0; i<ni; i++) {
    for(size_t k=0; k<nk; k++) {
    for(size_t l=0; l<nl; l++) {
        ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
        abs_index<4> ac(ic, dimc);
        double cjikl = 0.0;
        for(size_t p=0; p<np; p++) {
        ia[0]=p; ia[1]=i;
        ib[0]=j; ib[1]=p; ib[2]=k; ib[3]=l;
        abs_index<2> aa(ia, dima);
        abs_index<4> ab(ib, dimb);
        cjikl += dta[aa.get_abs_index()]*
         dtb[ab.get_abs_index()];
        }
        dtc2[ac.get_abs_index()] += d*cjikl;
        if(fabs(cjikl) > cij_max) cij_max = fabs(cjikl);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc; permc.permute(0, 1);
    contraction2<1, 3, 1> contr(permc);
    contr.contract(0, 1);

    to_contract2<1, 3, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ijkl_ijp_klp(size_t ni, size_t nj,
    size_t nk, size_t nl, size_t np) {

    // c_{ijkl} = \sum_{p} a_{ijp} b_{klp}

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ijp_klp(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ")";
    std::string tns = tnss.str();

    libtensor::index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
    libtensor::index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
    libtensor::index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
    index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
    index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
    index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

    // Generate reference data

    libtensor::index<3> ia, ib; libtensor::index<4> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
    for(size_t k=0; k<nk; k++) {
    for(size_t l=0; l<nl; l++) {
        ic[0]=i; ic[1]=j; ic[2]=k; ic[3]=l;
        abs_index<4> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        ia[0]=i; ia[1]=j; ia[2]=p;
        ib[0]=k; ib[1]=l; ib[2]=p;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        dtc2[ac.get_abs_index()] = cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    contraction2<2, 2, 1> contr(permc);
    contr.contract(2, 2);

    to_contract2<2, 2, 1, double>(contr, ta, tb, 1.0).perform(true, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ijkl_ijp_klp_a(size_t ni, size_t nj, size_t nk,
    size_t nl, size_t np, double d) {

    // c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ijp_klp_a(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
    std::string tns = tnss.str();

    libtensor::index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
    libtensor::index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
    libtensor::index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
    index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
    index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
    index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<3, double, allocator> ta(dima);
    dense_tensor<3, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cij_max = 0.0;

    {
    dense_tensor_ctrl<3, double> tca(ta);
    dense_tensor_ctrl<3, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    // Fill in random input

    for(size_t i=0; i<sza; i++) dta[i]=drand48();
    for(size_t i=0; i<szb; i++) dtb[i]=drand48();
    for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

    // Generate reference data

    libtensor::index<3> ia, ib; libtensor::index<4> ic;
    for(size_t i=0; i<ni; i++) {
    for(size_t j=0; j<nj; j++) {
    for(size_t k=0; k<nk; k++) {
    for(size_t l=0; l<nl; l++) {
        ic[0]=i; ic[1]=j; ic[2]=k; ic[3]=l;
        abs_index<4> ac(ic, dimc);
        double cij = 0.0;
        for(size_t p=0; p<np; p++) {
        ia[0]=i; ia[1]=j; ia[2]=p;
        ib[0]=k; ib[1]=l; ib[2]=p;
        abs_index<3> aa(ia, dima), ab(ib, dimb);
        cij += dta[aa.get_abs_index()]*dtb[ab.get_abs_index()];
        }
        dtc2[ac.get_abs_index()] += d*cij;
        if(fabs(cij) > cij_max) cij_max = fabs(cij);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    // Invoke the contraction routine

    permutation<4> permc;
    contraction2<2, 2, 1> contr(permc);
    contr.contract(2, 2);

    to_contract2<2, 2, 1, double>(contr, ta, tb, d).perform(false, tc);

    // Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);

    return 0;
}


int test_ijkl_ij_kl(size_t ni, size_t nj, size_t nk, size_t nl) {

    //
    // c_{ijkl} = a_{ij} b_{kl}
    //

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ij_kl(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ")";

    try {

    libtensor::index<2> ia1, ia2;
    libtensor::index<2> ib1, ib2;
    libtensor::index<4> ic1, ic2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cijkl_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //
    // Fill in random input
    //

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();

    //
    // Generate reference data
    //

    libtensor::index<2> ia, ib; libtensor::index<4> ic;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {

        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        ia[0] = i; ia[1] = j;
        ib[0] = k; ib[1] = l;
        abs_index<2> aia(ia, dima);
        abs_index<2> aib(ib, dimb);
        abs_index<4> aic(ic, dimc);
        double cijkl = dta[aia.get_abs_index()]*
        dtb[aib.get_abs_index()];
        dtc2[aic.get_abs_index()] = cijkl;
        if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    //
    // Invoke the contraction routine
    //

    contraction2<2, 2, 0> contr;

    to_contract2<2, 2, 0, double>(contr, ta, tb, 1.0).perform(true, tc);

    //
    // Compare against the reference
    //

    compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
        cijkl_max*k_thresh);

    } catch(exception &e) {
        return fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ijkl_ij_lk(size_t ni, size_t nj, size_t nk, size_t nl) {

    //
    // c_{ijkl} = a_{ij} b_{lk}
    //

    std::ostringstream tnss;
    tnss << "tod_contract2_test::test_ijkl_ij_lk(" << ni << ", " << nj
        << ", " << nk << ", " << nl << ")";

    try {

    libtensor::index<2> ia1, ia2;
    libtensor::index<2> ib1, ib2;
    libtensor::index<4> ic1, ic2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    ib2[0] = nl - 1; ib2[1] = nk - 1;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    size_t sza = dima.get_size(), szb = dimb.get_size(),
        szc = dimc.get_size();

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    double cijkl_max = 0.0;

    {
    dense_tensor_ctrl<2, double> tca(ta);
    dense_tensor_ctrl<2, double> tcb(tb);
    dense_tensor_ctrl<4, double> tcc(tc);
    dense_tensor_ctrl<4, double> tcc_ref(tc_ref);
    double *dta = tca.req_dataptr();
    double *dtb = tcb.req_dataptr();
    double *dtc1 = tcc.req_dataptr();
    double *dtc2 = tcc_ref.req_dataptr();

    //
    // Fill in random input
    //

    for(size_t i = 0; i < sza; i++) dta[i] = drand48();
    for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
    for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();

    //
    // Generate reference data
    //

    libtensor::index<2> ia, ib; libtensor::index<4> ic;
    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
    for(size_t k = 0; k < nk; k++) {
    for(size_t l = 0; l < nl; l++) {

        ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
        ia[0] = i; ia[1] = j;
        ib[0] = l; ib[1] = k;
        abs_index<2> aia(ia, dima);
        abs_index<2> aib(ib, dimb);
        abs_index<4> aic(ic, dimc);
        double cijkl = dta[aia.get_abs_index()]*
        dtb[aib.get_abs_index()];
        dtc2[aic.get_abs_index()] = cijkl;
        if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
    }
    }
    }
    }

    tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
    tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
    tcc.ret_dataptr(dtc1); dtc1 = NULL;
    tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
    }

    //
    // Invoke the contraction routine
    //

    permutation<4> permc;
    permc.permute(2, 3);
    contraction2<2, 2, 0> contr(permc);

    to_contract2<2, 2, 0, double>(contr, ta, tb, 1.0).perform(true, tc);

    //
    // Compare against the reference
    //

    compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
        cijkl_max*k_thresh);

    } catch(exception &e) {
        return fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    int rc = 1;
    allocator::init(16, 16, 16777216, 16777216);

    try {

    rc =

    //
    // Test one-index contractions
    //

// test_0_p_p(1);
// test_0_p_p(2);
// test_0_p_p(5);
// test_0_p_p(16);
// test_0_p_p(1, -0.5);
// test_0_p_p(2, -2.0);
// test_0_p_p(5, 1.2);
// test_0_p_p(16, 0.7);

    test_i_p_pi(1, 1, 0.0) |
    test_i_p_pi(1, 2, 0.0) |
    test_i_p_pi(2, 1, 0.0) |
    test_i_p_pi(3, 3, 0.0) |
    test_i_p_pi(3, 5, 0.0) |
    test_i_p_pi(16, 16, 0.0) |
    test_i_p_pi(1, 1, -0.5) |
    test_i_p_pi(1, 2, 2.0) |
    test_i_p_pi(2, 1, -1.0) |
    test_i_p_pi(3, 3, 3.7) |
    test_i_p_pi(3, 5, 1.0) |
    test_i_p_pi(16, 16, 0.7) |

    test_i_p_ip(1, 1, 0.0) |
    test_i_p_ip(1, 2, 0.0) |
    test_i_p_ip(2, 1, 0.0) |
    test_i_p_ip(3, 3, 0.0) |
    test_i_p_ip(3, 5, 0.0) |
    test_i_p_ip(16, 16, 0.0) |
    test_i_p_ip(1, 1, -0.5) |
    test_i_p_ip(1, 2, 2.0) |
    test_i_p_ip(2, 1, -1.0) |
    test_i_p_ip(3, 3, 3.7) |
    test_i_p_ip(3, 5, 1.0) |
    test_i_p_ip(16, 16, 0.7) |

    test_i_pi_p(1, 1, 0.0) |
    test_i_pi_p(1, 2, 0.0) |
    test_i_pi_p(2, 1, 0.0) |
    test_i_pi_p(3, 3, 0.0) |
    test_i_pi_p(3, 5, 0.0) |
    test_i_pi_p(16, 16, 0.0) |
    test_i_pi_p(1, 1, -0.5) |
    test_i_pi_p(1, 2, 2.0) |
    test_i_pi_p(2, 1, -1.0) |
    test_i_pi_p(3, 3, 3.7) |
    test_i_pi_p(3, 5, 1.0) |
    test_i_pi_p(16, 16, 0.7) |

    test_i_ip_p(1, 1, 0.0) |
    test_i_ip_p(1, 2, 0.0) |
    test_i_ip_p(2, 1, 0.0) |
    test_i_ip_p(3, 3, 0.0) |
    test_i_ip_p(3, 5, 0.0) |
    test_i_ip_p(16, 16, 0.0) |
    test_i_ip_p(1, 1, -0.5) |
    test_i_ip_p(1, 2, 2.0) |
    test_i_ip_p(2, 1, -1.0) |
    test_i_ip_p(3, 3, 3.7) |
    test_i_ip_p(3, 5, 1.0) |
    test_i_ip_p(16, 16, 0.7) |

    test_ij_i_j(1, 1, 0.0) |
    test_ij_i_j(1, 2, 0.0) |
    test_ij_i_j(2, 1, 0.0) |
    test_ij_i_j(3, 3, 0.0) |
    test_ij_i_j(3, 5, 0.0) |
    test_ij_i_j(16, 16, 0.0) |
    test_ij_i_j(1, 1, -0.5) |
    test_ij_i_j(1, 2, 2.0) |
    test_ij_i_j(2, 1, -1.0) |
    test_ij_i_j(3, 3, 3.7) |
    test_ij_i_j(3, 5, 1.0) |
    test_ij_i_j(16, 16, 0.7) |

    test_ij_j_i(1, 1, 0.0) |
    test_ij_j_i(1, 2, 0.0) |
    test_ij_j_i(2, 1, 0.0) |
    test_ij_j_i(3, 3, 0.0) |
    test_ij_j_i(3, 5, 0.0) |
    test_ij_j_i(16, 16, 0.0) |
    test_ij_j_i(1, 1, -0.5) |
    test_ij_j_i(1, 2, 2.0) |
    test_ij_j_i(2, 1, -1.0) |
    test_ij_j_i(3, 3, 3.7) |
    test_ij_j_i(3, 5, 1.0) |
    test_ij_j_i(16, 16, 0.7) |

    test_ij_pi_pj(1, 1, 1, 0.0) |
    test_ij_pi_pj(1, 1, 2, 0.0) |
    test_ij_pi_pj(1, 2, 1, 0.0) |
    test_ij_pi_pj(2, 1, 1, 0.0) |
    test_ij_pi_pj(3, 3, 3, 0.0) |
    test_ij_pi_pj(3, 5, 7, 0.0) |
    test_ij_pi_pj(16, 16, 16, 0.0) |
    test_ij_pi_pj(1, 1, 1, -0.5) |
    test_ij_pi_pj(1, 1, 2, 2.0) |
    test_ij_pi_pj(1, 2, 1, -1.0) |
    test_ij_pi_pj(2, 1, 1, 3.7) |
    test_ij_pi_pj(3, 3, 3, 1.0) |
    test_ij_pi_pj(3, 5, 7, -1.2) |
    test_ij_pi_pj(16, 16, 16, 0.7) |

    test_ij_pi_jp(1, 1, 1, 0.0) |
    test_ij_pi_jp(1, 1, 2, 0.0) |
    test_ij_pi_jp(1, 2, 1, 0.0) |
    test_ij_pi_jp(2, 1, 1, 0.0) |
    test_ij_pi_jp(3, 3, 3, 0.0) |
    test_ij_pi_jp(3, 5, 7, 0.0) |
    test_ij_pi_jp(16, 16, 16, 0.0) |
    test_ij_pi_jp(1, 1, 1, -0.5) |
    test_ij_pi_jp(1, 1, 2, 2.0) |
    test_ij_pi_jp(1, 2, 1, -1.0) |
    test_ij_pi_jp(2, 1, 1, 3.7) |
    test_ij_pi_jp(3, 3, 3, 1.0) |
    test_ij_pi_jp(3, 5, 7, -1.2) |
    test_ij_pi_jp(16, 16, 16, 0.7) |

    test_ij_ip_pj(1, 1, 1, 0.0) |
    test_ij_ip_pj(1, 1, 2, 0.0) |
    test_ij_ip_pj(1, 2, 1, 0.0) |
    test_ij_ip_pj(2, 1, 1, 0.0) |
    test_ij_ip_pj(3, 3, 3, 0.0) |
    test_ij_ip_pj(3, 5, 7, 0.0) |
    test_ij_ip_pj(16, 16, 16, 0.0) |
    test_ij_ip_pj(1, 1, 1, -0.5) |
    test_ij_ip_pj(1, 1, 2, 2.0) |
    test_ij_ip_pj(1, 2, 1, -1.0) |
    test_ij_ip_pj(2, 1, 1, 3.7) |
    test_ij_ip_pj(3, 3, 3, 1.0) |
    test_ij_ip_pj(3, 5, 7, -1.2) |
    test_ij_ip_pj(16, 16, 16, 0.7) |

    test_ij_ip_jp(1, 1, 1, 0.0) |
    test_ij_ip_jp(1, 1, 2, 0.0) |
    test_ij_ip_jp(1, 2, 1, 0.0) |
    test_ij_ip_jp(2, 1, 1, 0.0) |
    test_ij_ip_jp(3, 3, 3, 0.0) |
    test_ij_ip_jp(3, 5, 7, 0.0) |
    test_ij_ip_jp(16, 16, 16, 0.0) |
    test_ij_ip_jp(1, 1, 1, -0.5) |
    test_ij_ip_jp(1, 1, 2, 2.0) |
    test_ij_ip_jp(1, 2, 1, -1.0) |
    test_ij_ip_jp(2, 1, 1, 3.7) |
    test_ij_ip_jp(3, 3, 3, 1.0) |
    test_ij_ip_jp(3, 5, 7, -1.2) |
    test_ij_ip_jp(16, 16, 16, 0.7) |

    test_ij_pj_pi(1, 1, 1, 0.0) |
    test_ij_pj_pi(1, 1, 2, 0.0) |
    test_ij_pj_pi(1, 2, 1, 0.0) |
    test_ij_pj_pi(2, 1, 1, 0.0) |
    test_ij_pj_pi(3, 3, 3, 0.0) |
    test_ij_pj_pi(3, 5, 7, 0.0) |
    test_ij_pj_pi(16, 16, 16, 0.0) |
    test_ij_pj_pi(1, 1, 1, -0.5) |
    test_ij_pj_pi(1, 1, 2, 2.0) |
    test_ij_pj_pi(1, 2, 1, -1.0) |
    test_ij_pj_pi(2, 1, 1, 3.7) |
    test_ij_pj_pi(3, 3, 3, 1.0) |
    test_ij_pj_pi(3, 5, 7, -1.2) |
    test_ij_pj_pi(16, 16, 16, 0.7) |

    test_ij_pj_ip(1, 1, 1, 0.0) |
    test_ij_pj_ip(1, 1, 2, 0.0) |
    test_ij_pj_ip(1, 2, 1, 0.0) |
    test_ij_pj_ip(2, 1, 1, 0.0) |
    test_ij_pj_ip(3, 3, 3, 0.0) |
    test_ij_pj_ip(3, 5, 7, 0.0) |
    test_ij_pj_ip(16, 16, 16, 0.0) |
    test_ij_pj_ip(1, 1, 1, -0.5) |
    test_ij_pj_ip(1, 1, 2, 2.0) |
    test_ij_pj_ip(1, 2, 1, -1.0) |
    test_ij_pj_ip(2, 1, 1, 3.7) |
    test_ij_pj_ip(3, 3, 3, 1.0) |
    test_ij_pj_ip(3, 5, 7, -1.2) |
    test_ij_pj_ip(16, 16, 16, 0.7) |

    test_ij_jp_ip(1, 1, 1, 0.0) |
    test_ij_jp_ip(1, 1, 2, 0.0) |
    test_ij_jp_ip(1, 2, 1, 0.0) |
    test_ij_jp_ip(2, 1, 1, 0.0) |
    test_ij_jp_ip(3, 3, 3, 0.0) |
    test_ij_jp_ip(3, 5, 7, 0.0) |
    test_ij_jp_ip(16, 16, 16, 0.0) |
    test_ij_jp_ip(1, 1, 1, -0.5) |
    test_ij_jp_ip(1, 1, 2, 2.0) |
    test_ij_jp_ip(1, 2, 1, -1.0) |
    test_ij_jp_ip(2, 1, 1, 3.7) |
    test_ij_jp_ip(3, 3, 3, 1.0) |
    test_ij_jp_ip(3, 5, 7, -1.2) |
    test_ij_jp_ip(16, 16, 16, 0.7) |

    test_ij_jp_pi(1, 1, 1, 0.0) |
    test_ij_jp_pi(1, 1, 2, 0.0) |
    test_ij_jp_pi(1, 2, 1, 0.0) |
    test_ij_jp_pi(2, 1, 1, 0.0) |
    test_ij_jp_pi(3, 3, 3, 0.0) |
    test_ij_jp_pi(3, 5, 7, 0.0) |
    test_ij_jp_pi(16, 16, 16, 0.0) |
    test_ij_jp_pi(1, 1, 1, -0.5) |
    test_ij_jp_pi(1, 1, 2, 2.0) |
    test_ij_jp_pi(1, 2, 1, -1.0) |
    test_ij_jp_pi(2, 1, 1, 3.7) |
    test_ij_jp_pi(3, 3, 3, 1.0) |
    test_ij_jp_pi(3, 5, 7, -1.2) |
    test_ij_jp_pi(16, 16, 16, 0.7) |

    test_ij_p_pji(1, 1, 1, 0.0) |
    test_ij_p_pji(1, 1, 2, 0.0) |
    test_ij_p_pji(1, 2, 1, 0.0) |
    test_ij_p_pji(2, 1, 1, 0.0) |
    test_ij_p_pji(3, 3, 3, 0.0) |
    test_ij_p_pji(3, 5, 7, 0.0) |
    test_ij_p_pji(16, 16, 16, 0.0) |
    test_ij_p_pji(1, 1, 1, -0.5) |
    test_ij_p_pji(1, 1, 2, 2.0) |
    test_ij_p_pji(1, 2, 1, -1.0) |
    test_ij_p_pji(2, 1, 1, 3.7) |
    test_ij_p_pji(3, 3, 3, 1.0) |
    test_ij_p_pji(3, 5, 7, -1.2) |
    test_ij_p_pji(16, 16, 16, 0.7) |

    test_ij_pji_p(1, 1, 1, 0.0) |
    test_ij_pji_p(1, 1, 2, 0.0) |
    test_ij_pji_p(1, 2, 1, 0.0) |
    test_ij_pji_p(2, 1, 1, 0.0) |
    test_ij_pji_p(3, 3, 3, 0.0) |
    test_ij_pji_p(3, 5, 7, 0.0) |
    test_ij_pji_p(16, 16, 16, 0.0) |
    test_ij_pji_p(1, 1, 1, -0.5) |
    test_ij_pji_p(1, 1, 2, 2.0) |
    test_ij_pji_p(1, 2, 1, -1.0) |
    test_ij_pji_p(2, 1, 1, 3.7) |
    test_ij_pji_p(3, 3, 3, 1.0) |
    test_ij_pji_p(3, 5, 7, -1.2) |
    test_ij_pji_p(16, 16, 16, 0.7) |

    test_ij_pi_pj_qi_jq(1, 1, 1, 1, 0.0) |
    test_ij_pi_pj_qi_jq(1, 1, 2, 1, 0.0) |
    test_ij_pi_pj_qi_jq(1, 2, 1, 3, 0.0) |
    test_ij_pi_pj_qi_jq(2, 1, 1, 10, 0.0) |
    test_ij_pi_pj_qi_jq(3, 3, 3, 3, 0.0) |
    test_ij_pi_pj_qi_jq(3, 5, 7, 11, 0.0) |
    test_ij_pi_pj_qi_jq(16, 16, 16, 16, 0.0) |
    test_ij_pi_pj_qi_jq(1, 1, 1, 1, -0.5) |
    test_ij_pi_pj_qi_jq(1, 1, 2, 1, 2.0) |
    test_ij_pi_pj_qi_jq(1, 2, 1, 3, -1.0) |
    test_ij_pi_pj_qi_jq(2, 1, 1, 10, 3.7) |
    test_ij_pi_pj_qi_jq(3, 3, 3, 3, 1.0) |
    test_ij_pi_pj_qi_jq(3, 5, 7, 11, -1.2) |
    test_ij_pi_pj_qi_jq(16, 16, 16, 16, 0.7) |

    test_ij_pi_pj_qi_qj(1, 1, 1, 1, 0.0) |
    test_ij_pi_pj_qi_qj(1, 1, 2, 1, 0.0) |
    test_ij_pi_pj_qi_qj(1, 2, 1, 3, 0.0) |
    test_ij_pi_pj_qi_qj(2, 1, 1, 10, 0.0) |
    test_ij_pi_pj_qi_qj(3, 3, 3, 3, 0.0) |
    test_ij_pi_pj_qi_qj(3, 5, 7, 11, 0.0) |
    test_ij_pi_pj_qi_qj(16, 16, 16, 16, 0.0) |
    test_ij_pi_pj_qi_qj(1, 1, 1, 1, -0.5) |
    test_ij_pi_pj_qi_qj(1, 1, 2, 1, 2.0) |
    test_ij_pi_pj_qi_qj(1, 2, 1, 3, -1.0) |
    test_ij_pi_pj_qi_qj(2, 1, 1, 10, 3.7) |
    test_ij_pi_pj_qi_qj(3, 3, 3, 3, 1.0) |
    test_ij_pi_pj_qi_qj(3, 5, 7, 11, -1.2) |
    test_ij_pi_pj_qi_qj(16, 16, 16, 16, 0.7) |

    test_ijk_ip_pkj(1, 1, 1, 1, 0.0) |
    test_ijk_ip_pkj(1, 1, 2, 1, 0.0) |
    test_ijk_ip_pkj(1, 2, 1, 2, 0.0) |
    test_ijk_ip_pkj(2, 1, 1, 3, 0.0) |
    test_ijk_ip_pkj(3, 3, 3, 3, 0.0) |
    test_ijk_ip_pkj(3, 5, 7, 11, 0.0) |
    test_ijk_ip_pkj(16, 16, 16, 16, 0.0) |
    test_ijk_ip_pkj(1, 1, 1, 1, -0.5) |
    test_ijk_ip_pkj(1, 1, 2, 1, 2.0) |
    test_ijk_ip_pkj(1, 2, 1, 2, -1.0) |
    test_ijk_ip_pkj(2, 1, 1, 3, 3.7) |
    test_ijk_ip_pkj(3, 3, 3, 3, 1.0) |
    test_ijk_ip_pkj(3, 5, 7, 11, -1.2) |
    test_ijk_ip_pkj(16, 16, 16, 16, 0.7) |

    test_ijk_pi_pkj(1, 1, 1, 1, 0.0) |
    test_ijk_pi_pkj(1, 1, 2, 1, 0.0) |
    test_ijk_pi_pkj(1, 2, 1, 2, 0.0) |
    test_ijk_pi_pkj(2, 1, 1, 3, 0.0) |
    test_ijk_pi_pkj(3, 3, 3, 3, 0.0) |
    test_ijk_pi_pkj(3, 5, 7, 11, 0.0) |
    test_ijk_pi_pkj(16, 16, 16, 16, 0.0) |
    test_ijk_pi_pkj(1, 1, 1, 1, -0.5) |
    test_ijk_pi_pkj(1, 1, 2, 1, 2.0) |
    test_ijk_pi_pkj(1, 2, 1, 2, -1.0) |
    test_ijk_pi_pkj(2, 1, 1, 3, 3.7) |
    test_ijk_pi_pkj(3, 3, 3, 3, 1.0) |
    test_ijk_pi_pkj(3, 5, 7, 11, -1.2) |
    test_ijk_pi_pkj(16, 16, 16, 16, 0.7) |

    test_ijk_pik_pj(1, 1, 1, 1, 0.0) |
    test_ijk_pik_pj(1, 1, 2, 1, 0.0) |
    test_ijk_pik_pj(1, 2, 1, 2, 0.0) |
    test_ijk_pik_pj(2, 1, 1, 3, 0.0) |
    test_ijk_pik_pj(3, 3, 3, 3, 0.0) |
    test_ijk_pik_pj(3, 5, 7, 11, 0.0) |
    test_ijk_pik_pj(16, 16, 16, 16, 0.0) |
    test_ijk_pik_pj(1, 1, 1, 1, -0.5) |
    test_ijk_pik_pj(1, 1, 2, 1, 2.0) |
    test_ijk_pik_pj(1, 2, 1, 2, -1.0) |
    test_ijk_pik_pj(2, 1, 1, 3, 3.7) |
    test_ijk_pik_pj(3, 3, 3, 3, 1.0) |
    test_ijk_pik_pj(3, 5, 7, 11, -1.2) |
    test_ijk_pik_pj(16, 16, 16, 16, 0.7) |

    test_ijk_pj_ipk(1, 1, 1, 1, 0.0) |
    test_ijk_pj_ipk(1, 1, 2, 1, 0.0) |
    test_ijk_pj_ipk(1, 2, 1, 2, 0.0) |
    test_ijk_pj_ipk(2, 1, 1, 3, 0.0) |
    test_ijk_pj_ipk(3, 3, 3, 3, 0.0) |
    test_ijk_pj_ipk(3, 5, 7, 11, 0.0) |
    test_ijk_pj_ipk(16, 16, 16, 16, 0.0) |
    test_ijk_pj_ipk(1, 1, 1, 1, -0.5) |
    test_ijk_pj_ipk(1, 1, 2, 1, 2.0) |
    test_ijk_pj_ipk(1, 2, 1, 2, -1.0) |
    test_ijk_pj_ipk(2, 1, 1, 3, 3.7) |
    test_ijk_pj_ipk(3, 3, 3, 3, 1.0) |
    test_ijk_pj_ipk(3, 5, 7, 11, -1.2) |
    test_ijk_pj_ipk(16, 16, 16, 16, 0.7) |

    test_ijk_pj_pik(1, 1, 1, 1, 0.0) |
    test_ijk_pj_pik(1, 1, 2, 1, 0.0) |
    test_ijk_pj_pik(1, 2, 1, 2, 0.0) |
    test_ijk_pj_pik(2, 1, 1, 3, 0.0) |
    test_ijk_pj_pik(3, 3, 3, 3, 0.0) |
    test_ijk_pj_pik(3, 5, 7, 11, 0.0) |
    test_ijk_pj_pik(16, 16, 16, 16, 0.0) |
    test_ijk_pj_pik(1, 1, 1, 1, -0.5) |
    test_ijk_pj_pik(1, 1, 2, 1, 2.0) |
    test_ijk_pj_pik(1, 2, 1, 2, -1.0) |
    test_ijk_pj_pik(2, 1, 1, 3, 3.7) |
    test_ijk_pj_pik(3, 3, 3, 3, 1.0) |
    test_ijk_pj_pik(3, 5, 7, 11, -1.2) |
    test_ijk_pj_pik(16, 16, 16, 16, 0.7) |

    test_ijk_pkj_ip(1, 1, 1, 1, 0.0) |
    test_ijk_pkj_ip(1, 1, 2, 1, 0.0) |
    test_ijk_pkj_ip(1, 2, 1, 2, 0.0) |
    test_ijk_pkj_ip(2, 1, 1, 3, 0.0) |
    test_ijk_pkj_ip(3, 3, 3, 3, 0.0) |
    test_ijk_pkj_ip(3, 5, 7, 11, 0.0) |
    test_ijk_pkj_ip(16, 16, 16, 16, 0.0) |
    test_ijk_pkj_ip(1, 1, 1, 1, -0.5) |
    test_ijk_pkj_ip(1, 1, 2, 1, 2.0) |
    test_ijk_pkj_ip(1, 2, 1, 2, -1.0) |
    test_ijk_pkj_ip(2, 1, 1, 3, 3.7) |
    test_ijk_pkj_ip(3, 3, 3, 3, 1.0) |
    test_ijk_pkj_ip(3, 5, 7, 11, -1.2) |
    test_ijk_pkj_ip(16, 16, 16, 16, 0.7) |

    test_ijk_pkj_pi(1, 1, 1, 1, 0.0) |
    test_ijk_pkj_pi(1, 1, 2, 1, 0.0) |
    test_ijk_pkj_pi(1, 2, 1, 2, 0.0) |
    test_ijk_pkj_pi(2, 1, 1, 3, 0.0) |
    test_ijk_pkj_pi(3, 3, 3, 3, 0.0) |
    test_ijk_pkj_pi(3, 5, 7, 11, 0.0) |
    test_ijk_pkj_pi(16, 16, 16, 16, 0.0) |
    test_ijk_pkj_pi(1, 1, 1, 1, -0.5) |
    test_ijk_pkj_pi(1, 1, 2, 1, 2.0) |
    test_ijk_pkj_pi(1, 2, 1, 2, -1.0) |
    test_ijk_pkj_pi(2, 1, 1, 3, 3.7) |
    test_ijk_pkj_pi(3, 3, 3, 3, 1.0) |
    test_ijk_pkj_pi(3, 5, 7, 11, -1.2) |
    test_ijk_pkj_pi(16, 16, 16, 16, 0.7) |

    test_ijkl_ikp_jpl(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ikp_jpl(2, 1, 1, 1, 1, 0.0) |
    test_ijkl_ikp_jpl(1, 2, 1, 1, 1, 0.0) |
    test_ijkl_ikp_jpl(1, 1, 2, 1, 1, 0.0) |
    test_ijkl_ikp_jpl(1, 1, 1, 2, 1, 0.0) |
    test_ijkl_ikp_jpl(1, 1, 1, 1, 2, 0.0) |
    test_ijkl_ikp_jpl(2, 3, 2, 3, 2, 0.0) |
    test_ijkl_ikp_jpl(3, 5, 1, 7, 13, 0.0) |
    test_ijkl_ikp_jpl(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ikp_jpl(1, 1, 1, 1, 1, -0.5) |
    test_ijkl_ikp_jpl(2, 1, 1, 1, 1, 2.0) |
    test_ijkl_ikp_jpl(1, 2, 1, 1, 1, -1.0) |
    test_ijkl_ikp_jpl(1, 1, 2, 1, 1, 3.7) |
    test_ijkl_ikp_jpl(1, 1, 1, 2, 1, 1.0) |
    test_ijkl_ikp_jpl(1, 1, 1, 1, 2, -1.2) |
    test_ijkl_ikp_jpl(2, 3, 2, 3, 2, 12.3) |
    test_ijkl_ikp_jpl(3, 5, 1, 7, 13, -1.25) |

    test_ijkl_ipk_jpl(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipk_jpl(2, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipk_jpl(1, 2, 1, 1, 1, 0.0) |
    test_ijkl_ipk_jpl(1, 1, 2, 1, 1, 0.0) |
    test_ijkl_ipk_jpl(1, 1, 1, 2, 1, 0.0) |
    test_ijkl_ipk_jpl(1, 1, 1, 1, 2, 0.0) |
    test_ijkl_ipk_jpl(2, 3, 2, 3, 2, 0.0) |
    test_ijkl_ipk_jpl(3, 5, 1, 7, 13, 0.0) |
    test_ijkl_ipk_jpl(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipk_jpl(1, 1, 1, 1, 1, -0.5) |
    test_ijkl_ipk_jpl(2, 1, 1, 1, 1, 2.0) |
    test_ijkl_ipk_jpl(1, 2, 1, 1, 1, -1.0) |
    test_ijkl_ipk_jpl(1, 1, 2, 1, 1, 3.7) |
    test_ijkl_ipk_jpl(1, 1, 1, 2, 1, 1.0) |
    test_ijkl_ipk_jpl(1, 1, 1, 1, 2, -1.2) |
    test_ijkl_ipk_jpl(2, 3, 2, 3, 2, 12.3) |
    test_ijkl_ipk_jpl(3, 5, 1, 7, 13, -1.25) |

    test_ijkl_ipl_jpk(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipl_jpk(2, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipl_jpk(1, 2, 1, 1, 1, 0.0) |
    test_ijkl_ipl_jpk(1, 1, 2, 1, 1, 0.0) |
    test_ijkl_ipl_jpk(1, 1, 1, 2, 1, 0.0) |
    test_ijkl_ipl_jpk(1, 1, 1, 1, 2, 0.0) |
    test_ijkl_ipl_jpk(2, 3, 2, 3, 2, 0.0) |
    test_ijkl_ipl_jpk(3, 5, 1, 7, 13, 0.0) |
    test_ijkl_ipl_jpk(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipl_jpk(1, 1, 1, 1, 1, -0.5) |
    test_ijkl_ipl_jpk(2, 1, 1, 1, 1, 2.0) |
    test_ijkl_ipl_jpk(1, 2, 1, 1, 1, -1.0) |
    test_ijkl_ipl_jpk(1, 1, 2, 1, 1, 3.7) |
    test_ijkl_ipl_jpk(1, 1, 1, 2, 1, 1.0) |
    test_ijkl_ipl_jpk(1, 1, 1, 1, 2, -1.2) |
    test_ijkl_ipl_jpk(2, 3, 2, 3, 2, 12.3) |
    test_ijkl_ipl_jpk(3, 5, 1, 7, 13, -1.25) |

    test_ijkl_jkp_ipl(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jkp_ipl(2, 1, 1, 1, 1, 0.0) |
    test_ijkl_jkp_ipl(1, 2, 1, 1, 1, 0.0) |
    test_ijkl_jkp_ipl(1, 1, 2, 1, 1, 0.0) |
    test_ijkl_jkp_ipl(1, 1, 1, 2, 1, 0.0) |
    test_ijkl_jkp_ipl(1, 1, 1, 1, 2, 0.0) |
    test_ijkl_jkp_ipl(2, 3, 2, 3, 2, 0.0) |
    test_ijkl_jkp_ipl(3, 5, 1, 7, 13, 0.0) |
    test_ijkl_jkp_ipl(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jkp_ipl(1, 1, 1, 1, 1, -0.5) |
    test_ijkl_jkp_ipl(2, 1, 1, 1, 1, 2.0) |
    test_ijkl_jkp_ipl(1, 2, 1, 1, 1, -1.0) |
    test_ijkl_jkp_ipl(1, 1, 2, 1, 1, 3.7) |
    test_ijkl_jkp_ipl(1, 1, 1, 2, 1, 1.0) |
    test_ijkl_jkp_ipl(1, 1, 1, 1, 2, -1.2) |
    test_ijkl_jkp_ipl(2, 3, 2, 3, 2, 12.3) |
    test_ijkl_jkp_ipl(3, 5, 1, 7, 13, -1.25) |

    test_ijkl_jpl_ipk(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk(2, 1, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk(1, 2, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk(1, 1, 2, 1, 1, 0.0) |
    test_ijkl_jpl_ipk(1, 1, 1, 2, 1, 0.0) |
    test_ijkl_jpl_ipk(1, 1, 1, 1, 2, 0.0) |
    test_ijkl_jpl_ipk(2, 3, 2, 3, 2, 0.0) |
    test_ijkl_jpl_ipk(3, 5, 1, 7, 13, 0.0) |
    test_ijkl_jpl_ipk(1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk(1, 1, 1, 1, 1, -0.5) |
    test_ijkl_jpl_ipk(2, 1, 1, 1, 1, 2.0) |
    test_ijkl_jpl_ipk(1, 2, 1, 1, 1, -1.0) |
    test_ijkl_jpl_ipk(1, 1, 2, 1, 1, 3.7) |
    test_ijkl_jpl_ipk(1, 1, 1, 2, 1, 1.0) |
    test_ijkl_jpl_ipk(1, 1, 1, 1, 2, -1.2) |
    test_ijkl_jpl_ipk(2, 3, 2, 3, 2, 12.3) |
    test_ijkl_jpl_ipk(3, 5, 1, 7, 13, -1.25) |

    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(2, 1, 1, 1, 1, 2, 3, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 2, 1, 1, 2, 2, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 2, 1, 1, 2, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 1, 2, 2, 1, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(2, 3, 2, 3, 2, 2, 3, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(3, 5, 1, 7, 13, 11, 3, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 1, 1, 2, 3, -0.5) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(2, 1, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 2, 1, 1, 1, 2, 2, -1.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 2, 1, 1, 1, 2, 3.7) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 2, 1, 2, 1, 1.0) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(1, 1, 1, 1, 2, 2, 1, -1.2) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(2, 3, 2, 3, 2, 2, 3, 12.3) |
    test_ijkl_jpl_ipk_jiq_kql_jlr_ikr(3, 5, 1, 7, 13, 11, 3, -1.25) |

    test_ijklm_ikp_jpml(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_ikp_jpml(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_ikp_jpml(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijklm_ikp_jpml(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijklm_ikp_jpml(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijklm_ikp_jpml(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijklm_ikp_jpml(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijklm_ikp_jpml(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijklm_ikp_jpml(3, 3, 3, 3, 3, 3, 0.0) |
    test_ijklm_ikp_jpml(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijklm_ikp_jpml(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_ikp_jpml(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijklm_ikp_jpml(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijklm_ikp_jpml(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijklm_ikp_jpml(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijklm_ikp_jpml(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijklm_ikp_jpml(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijklm_ikp_jpml(1, 1, 1, 1, 1, 2, 1.2) |
    test_ijklm_ikp_jpml(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijklm_ikp_jpml(3, 3, 3, 3, 3, 3, 12.3) |
    test_ijklm_ikp_jpml(3, 5, 1, 7, 13, 11, -1.25) |

    test_ijklm_ipkm_jpl(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_ipkm_jpl(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_ipkm_jpl(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijklm_ipkm_jpl(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijklm_ipkm_jpl(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijklm_ipkm_jpl(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijklm_ipkm_jpl(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijklm_ipkm_jpl(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijklm_ipkm_jpl(3, 3, 3, 3, 3, 3, 0.0) |
    test_ijklm_ipkm_jpl(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijklm_ipkm_jpl(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_ipkm_jpl(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijklm_ipkm_jpl(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijklm_ipkm_jpl(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijklm_ipkm_jpl(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijklm_ipkm_jpl(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijklm_ipkm_jpl(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijklm_ipkm_jpl(1, 1, 1, 1, 1, 2, 1.2) |
    test_ijklm_ipkm_jpl(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijklm_ipkm_jpl(3, 3, 3, 3, 3, 3, 12.3) |
    test_ijklm_ipkm_jpl(3, 5, 1, 7, 13, 11, -1.25) |

    test_ijklm_jlp_ipkm(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_jlp_ipkm(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_jlp_ipkm(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijklm_jlp_ipkm(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijklm_jlp_ipkm(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijklm_jlp_ipkm(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijklm_jlp_ipkm(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijklm_jlp_ipkm(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijklm_jlp_ipkm(3, 3, 3, 3, 3, 3, 0.0) |
    test_ijklm_jlp_ipkm(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijklm_jlp_ipkm(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklm_jlp_ipkm(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijklm_jlp_ipkm(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijklm_jlp_ipkm(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijklm_jlp_ipkm(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijklm_jlp_ipkm(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijklm_jlp_ipkm(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijklm_jlp_ipkm(1, 1, 1, 1, 1, 2, 1.2) |
    test_ijklm_jlp_ipkm(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijklm_jlp_ipkm(3, 3, 3, 3, 3, 3, 12.3) |
    test_ijklm_jlp_ipkm(3, 5, 1, 7, 13, 11, -1.25) |

    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(2, 1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 2, 1, 1, 1, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 2, 1, 1, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 2, 1, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 2, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 2, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 1, 2, 0.0) |
    test_ijklmn_kjmp_ipln(2, 3, 2, 3, 2, 3, 2, 0.0) |
    test_ijklmn_kjmp_ipln(3, 3, 3, 3, 3, 3, 3, 0.0) |
    test_ijklmn_kjmp_ipln(3, 5, 1, 7, 13, 11, 17, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 1, 1, 0.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 1, 1, -0.5) |
    test_ijklmn_kjmp_ipln(2, 1, 1, 1, 1, 1, 1, 2.0) |
    test_ijklmn_kjmp_ipln(1, 2, 1, 1, 1, 1, 1, -1.0) |
    test_ijklmn_kjmp_ipln(1, 1, 2, 1, 1, 1, 1, 3.7) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 2, 1, 1, 1, 1.0) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 2, 1, 1, -1.2) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 2, 1, 1.2) |
    test_ijklmn_kjmp_ipln(1, 1, 1, 1, 1, 1, 2, -1.3) |
    test_ijklmn_kjmp_ipln(2, 3, 2, 3, 2, 3, 2, 12.3) |
    test_ijklmn_kjmp_ipln(3, 3, 3, 3, 3, 3, 3, 0.4) |
    test_ijklmn_kjmp_ipln(3, 5, 1, 7, 13, 11, 17, -1.25) |

    //
    // Test two-index contractions
    //

    test_ij_pqi_pjq(1, 1, 1, 1, 0.0) |
    test_ij_pqi_pjq(1, 1, 1, 2, 0.0) |
    test_ij_pqi_pjq(1, 1, 2, 1, 0.0) |
    test_ij_pqi_pjq(1, 2, 1, 1, 0.0) |
    test_ij_pqi_pjq(2, 1, 1, 1, 0.0) |
    test_ij_pqi_pjq(3, 3, 3, 3, 0.0) |
    test_ij_pqi_pjq(11, 5, 7, 3, 0.0) |
    test_ij_pqi_pjq(16, 16, 16, 16, 0.0) |
    test_ij_pqi_pjq(1, 1, 1, 1, -0.5) |
    test_ij_pqi_pjq(1, 1, 1, 2, 2.0) |
    test_ij_pqi_pjq(1, 1, 2, 1, -1.0) |
    test_ij_pqi_pjq(1, 2, 1, 1, 3.7) |
    test_ij_pqi_pjq(2, 1, 1, 1, 1.0) |
    test_ij_pqi_pjq(3, 3, 3, 3, 1.0) |
    test_ij_pqi_pjq(11, 5, 7, 3, -1.2) |
    test_ij_pqi_pjq(16, 16, 16, 16, 0.7) |

    test_ij_ipq_jqp(1, 1, 1, 1, 0.0) |
    test_ij_ipq_jqp(1, 1, 1, 2, 0.0) |
    test_ij_ipq_jqp(1, 1, 2, 1, 0.0) |
    test_ij_ipq_jqp(1, 2, 1, 1, 0.0) |
    test_ij_ipq_jqp(2, 1, 1, 1, 0.0) |
    test_ij_ipq_jqp(3, 3, 3, 3, 0.0) |
    test_ij_ipq_jqp(11, 5, 7, 3, 0.0) |
    test_ij_ipq_jqp(16, 16, 16, 16, 0.0) |
    test_ij_ipq_jqp(1, 1, 1, 1, -0.5) |
    test_ij_ipq_jqp(1, 1, 1, 2, 2.0) |
    test_ij_ipq_jqp(1, 1, 2, 1, -1.0) |
    test_ij_ipq_jqp(1, 2, 1, 1, 3.7) |
    test_ij_ipq_jqp(2, 1, 1, 1, 1.0) |
    test_ij_ipq_jqp(3, 3, 3, 3, 1.0) |
    test_ij_ipq_jqp(11, 5, 7, 3, -1.2) |
    test_ij_ipq_jqp(16, 16, 16, 16, 0.7) |

    test_ij_jpq_iqp(1, 1, 1, 1, 0.0) |
    test_ij_jpq_iqp(1, 1, 1, 2, 0.0) |
    test_ij_jpq_iqp(1, 1, 2, 1, 0.0) |
    test_ij_jpq_iqp(1, 2, 1, 1, 0.0) |
    test_ij_jpq_iqp(2, 1, 1, 1, 0.0) |
    test_ij_jpq_iqp(3, 3, 3, 3, 0.0) |
    test_ij_jpq_iqp(11, 5, 7, 3, 0.0) |
    test_ij_jpq_iqp(16, 16, 16, 16, 0.0) |
    test_ij_jpq_iqp(1, 1, 1, 1, -0.5) |
    test_ij_jpq_iqp(1, 1, 1, 2, 2.0) |
    test_ij_jpq_iqp(1, 1, 2, 1, -1.0) |
    test_ij_jpq_iqp(1, 2, 1, 1, 3.7) |
    test_ij_jpq_iqp(2, 1, 1, 1, 1.0) |
    test_ij_jpq_iqp(3, 3, 3, 3, 1.0) |
    test_ij_jpq_iqp(11, 5, 7, 3, -1.2) |
    test_ij_jpq_iqp(16, 16, 16, 16, 0.7) |

    test_ij_jipq_qp(1, 1, 1, 1, 0.0) |
    test_ij_jipq_qp(1, 1, 1, 2, 0.0) |
    test_ij_jipq_qp(1, 1, 2, 1, 0.0) |
    test_ij_jipq_qp(1, 2, 1, 1, 0.0) |
    test_ij_jipq_qp(2, 1, 1, 1, 0.0) |
    test_ij_jipq_qp(3, 3, 3, 3, 0.0) |
    test_ij_jipq_qp(11, 5, 7, 3, 0.0) |
    test_ij_jipq_qp(16, 16, 16, 16, 0.0) |
    test_ij_jipq_qp(1, 1, 1, 1, -0.5) |
    test_ij_jipq_qp(1, 1, 1, 2, 2.0) |
    test_ij_jipq_qp(1, 1, 2, 1, -1.0) |
    test_ij_jipq_qp(1, 2, 1, 1, 3.7) |
    test_ij_jipq_qp(2, 1, 1, 1, 1.0) |
    test_ij_jipq_qp(3, 3, 3, 3, 1.0) |
    test_ij_jipq_qp(11, 5, 7, 3, -1.2) |
    test_ij_jipq_qp(16, 16, 16, 16, 0.7) |

    test_ij_pq_ijpq(1, 1, 1, 1) |
    test_ij_pq_ijpq(2, 2, 2, 2) |
    test_ij_pq_ijpq_a(1, 1, 1, 1, 0.25) |
    test_ij_pq_ijpq_a(2, 2, 2, 2, 0.25) |

    test_ijk_pqj_iqpk(1, 1, 1, 1, 1, 0.0) |
    test_ijk_pqj_iqpk(2, 1, 1, 1, 1, 0.0) |
    test_ijk_pqj_iqpk(1, 2, 1, 1, 1, 0.0) |
    test_ijk_pqj_iqpk(1, 1, 2, 1, 1, 0.0) |
    test_ijk_pqj_iqpk(1, 1, 1, 2, 1, 0.0) |
    test_ijk_pqj_iqpk(1, 1, 1, 1, 2, 0.0) |
    test_ijk_pqj_iqpk(2, 3, 2, 3, 2, 0.0) |
    test_ijk_pqj_iqpk(3, 5, 1, 13, 11, 0.0) |
    test_ijk_pqj_iqpk(3, 5, 2, 13, 11, 0.0) |
    test_ijk_pqj_iqpk(1, 1, 1, 1, 1, 0.0) |
    test_ijk_pqj_iqpk(1, 1, 1, 1, 1, -0.5) |
    test_ijk_pqj_iqpk(2, 1, 1, 1, 1, 2.0) |
    test_ijk_pqj_iqpk(1, 2, 1, 1, 1, -1.0) |
    test_ijk_pqj_iqpk(1, 1, 2, 1, 1, 3.7) |
    test_ijk_pqj_iqpk(1, 1, 1, 2, 1, -1.2) |
    test_ijk_pqj_iqpk(1, 1, 1, 1, 2, 0.7) |
    test_ijk_pqj_iqpk(2, 3, 2, 2, 3, 12.3) |
    test_ijk_pqj_iqpk(3, 5, 1, 13, 11, -1.25) |
    test_ijk_pqj_iqpk(3, 5, 2, 13, 11, -1.25) |

    test_ijk_pqji_qpk(1, 1, 1, 1, 1, 0.0) |
    test_ijk_pqji_qpk(2, 1, 1, 1, 1, 0.0) |
    test_ijk_pqji_qpk(1, 2, 1, 1, 1, 0.0) |
    test_ijk_pqji_qpk(1, 1, 2, 1, 1, 0.0) |
    test_ijk_pqji_qpk(1, 1, 1, 2, 1, 0.0) |
    test_ijk_pqji_qpk(1, 1, 1, 1, 2, 0.0) |
    test_ijk_pqji_qpk(2, 3, 2, 3, 2, 0.0) |
    test_ijk_pqji_qpk(3, 5, 1, 13, 11, 0.0) |
    test_ijk_pqji_qpk(3, 5, 2, 13, 11, 0.0) |
    test_ijk_pqji_qpk(1, 1, 1, 1, 1, 0.0) |
    test_ijk_pqji_qpk(1, 1, 1, 1, 1, -0.5) |
    test_ijk_pqji_qpk(2, 1, 1, 1, 1, 2.0) |
    test_ijk_pqji_qpk(1, 2, 1, 1, 1, -1.0) |
    test_ijk_pqji_qpk(1, 1, 2, 1, 1, 3.7) |
    test_ijk_pqji_qpk(1, 1, 1, 2, 1, -1.2) |
    test_ijk_pqji_qpk(1, 1, 1, 1, 2, 0.7) |
    test_ijk_pqji_qpk(2, 3, 2, 2, 3, 12.3) |
    test_ijk_pqji_qpk(3, 5, 1, 13, 11, -1.25) |
    test_ijk_pqji_qpk(3, 5, 2, 13, 11, -1.25) |

    test_ijk_kjpq_iqp(1, 1, 1, 1, 1, 0.0) |
    test_ijk_kjpq_iqp(2, 1, 1, 1, 1, 0.0) |
    test_ijk_kjpq_iqp(1, 2, 1, 1, 1, 0.0) |
    test_ijk_kjpq_iqp(1, 1, 2, 1, 1, 0.0) |
    test_ijk_kjpq_iqp(1, 1, 1, 2, 1, 0.0) |
    test_ijk_kjpq_iqp(1, 1, 1, 1, 2, 0.0) |
    test_ijk_kjpq_iqp(2, 3, 2, 3, 2, 0.0) |
    test_ijk_kjpq_iqp(3, 5, 1, 13, 11, 0.0) |
    test_ijk_kjpq_iqp(3, 5, 2, 13, 11, 0.0) |
    test_ijk_kjpq_iqp(1, 1, 1, 1, 1, 0.0) |
    test_ijk_kjpq_iqp(1, 1, 1, 1, 1, -0.5) |
    test_ijk_kjpq_iqp(2, 1, 1, 1, 1, 2.0) |
    test_ijk_kjpq_iqp(1, 2, 1, 1, 1, -1.0) |
    test_ijk_kjpq_iqp(1, 1, 2, 1, 1, 3.7) |
    test_ijk_kjpq_iqp(1, 1, 1, 2, 1, -1.2) |
    test_ijk_kjpq_iqp(1, 1, 1, 1, 2, 0.7) |
    test_ijk_kjpq_iqp(2, 3, 2, 2, 3, 12.3) |
    test_ijk_kjpq_iqp(3, 5, 1, 13, 11, -1.25) |
    test_ijk_kjpq_iqp(3, 5, 2, 13, 11, -1.25) |

    test_ijk_pkiq_pjq(1, 1, 1, 1, 1, 0.0) |
    test_ijk_pkiq_pjq(2, 1, 1, 1, 1, 0.0) |
    test_ijk_pkiq_pjq(1, 2, 1, 1, 1, 0.0) |
    test_ijk_pkiq_pjq(1, 1, 2, 1, 1, 0.0) |
    test_ijk_pkiq_pjq(1, 1, 1, 2, 1, 0.0) |
    test_ijk_pkiq_pjq(1, 1, 1, 1, 2, 0.0) |
    test_ijk_pkiq_pjq(2, 3, 2, 3, 2, 0.0) |
    test_ijk_pkiq_pjq(3, 5, 1, 13, 11, 0.0) |
    test_ijk_pkiq_pjq(3, 5, 2, 13, 11, 0.0) |
    test_ijk_pkiq_pjq(1, 1, 1, 1, 1, 0.0) |
    test_ijk_pkiq_pjq(1, 1, 1, 1, 1, -0.5) |
    test_ijk_pkiq_pjq(2, 1, 1, 1, 1, 2.0) |
    test_ijk_pkiq_pjq(1, 2, 1, 1, 1, -1.0) |
    test_ijk_pkiq_pjq(1, 1, 2, 1, 1, 3.7) |
    test_ijk_pkiq_pjq(1, 1, 1, 2, 1, -1.2) |
    test_ijk_pkiq_pjq(1, 1, 1, 1, 2, 0.7) |
    test_ijk_pkiq_pjq(2, 3, 2, 2, 3, 12.3) |
    test_ijk_pkiq_pjq(3, 5, 1, 13, 11, -1.25) |
    test_ijk_pkiq_pjq(3, 5, 2, 13, 11, -1.25) |

    test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_kpjq(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_kpjq(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_kpjq(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_iplq_kpjq(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_iplq_kpjq(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_iplq_kpjq(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_iplq_kpjq(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_iplq_kpjq(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_iplq_kpjq(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_iplq_kpjq(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_iplq_kpjq(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_iplq_kpjq(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_iplq_kpjq(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_iplq_kpjq(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_iplq_kpjq(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_iplq_kpjq(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkjq(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkjq(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkjq(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkjq(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_iplq_pkjq(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_iplq_pkjq(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_iplq_pkjq(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_iplq_pkjq(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_iplq_pkjq(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_iplq_pkjq(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_iplq_pkjq(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_iplq_pkjq(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_iplq_pkjq(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_iplq_pkjq(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_iplq_pkjq(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_iplq_pkjq(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkqj(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkqj(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkqj(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkqj(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_iplq_pkqj(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_iplq_pkqj(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_iplq_pkqj(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_iplq_pkqj(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_iplq_pkqj(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_iplq_pkqj(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_iplq_pkqj(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_iplq_pkqj(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_iplq_pkqj(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_iplq_pkqj(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_iplq_pkqj(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_iplq_pkqj(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_kpqj(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_kpqj(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_kpqj(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_ipql_kpqj(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_ipql_kpqj(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_ipql_kpqj(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_ipql_kpqj(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_ipql_kpqj(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_ipql_kpqj(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_ipql_kpqj(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_ipql_kpqj(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_ipql_kpqj(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_ipql_kpqj(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_ipql_kpqj(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_ipql_kpqj(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_ipql_kpqj(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_pkqj(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_pkqj(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_pkqj(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_ipql_pkqj(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_ipql_pkqj(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_ipql_pkqj(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_ipql_pkqj(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_ipql_pkqj(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_ipql_pkqj(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_ipql_pkqj(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_ipql_pkqj(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_ipql_pkqj(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_ipql_pkqj(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_ipql_pkqj(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_ipql_pkqj(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_ipql_pkqj(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_kpjq(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_kpjq(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_kpjq(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_pilq_kpjq(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_pilq_kpjq(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_pilq_kpjq(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_pilq_kpjq(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_pilq_kpjq(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_pilq_kpjq(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_pilq_kpjq(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_pilq_kpjq(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_pilq_kpjq(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_pilq_kpjq(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_pilq_kpjq(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_pilq_kpjq(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_pilq_kpjq(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_pkjq(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_pkjq(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_pkjq(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_pilq_pkjq(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_pilq_pkjq(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_pilq_pkjq(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_pilq_pkjq(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_pilq_pkjq(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_pilq_pkjq(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_pilq_pkjq(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_pilq_pkjq(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_pilq_pkjq(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_pilq_pkjq(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_pilq_pkjq(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_pilq_pkjq(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_pilq_pkjq(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_kpqj(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_kpqj(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_kpqj(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_piql_kpqj(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_piql_kpqj(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_piql_kpqj(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_piql_kpqj(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_piql_kpqj(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_piql_kpqj(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_piql_kpqj(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_piql_kpqj(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_piql_kpqj(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_piql_kpqj(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_piql_kpqj(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_piql_kpqj(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_piql_kpqj(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_pkqj(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_pkqj(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_pkqj(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_piql_pkqj(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_piql_pkqj(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_piql_pkqj(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_piql_pkqj(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_piql_pkqj(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_piql_pkqj(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_piql_pkqj(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_piql_pkqj(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_piql_pkqj(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_piql_pkqj(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_piql_pkqj(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_piql_pkqj(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_piql_pkqj(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_iqpl(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_iqpl(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_iqpl(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_pqkj_iqpl(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_pqkj_iqpl(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_pqkj_iqpl(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_pqkj_iqpl(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_pqkj_iqpl(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_pqkj_iqpl(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_pqkj_iqpl(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_pqkj_iqpl(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_pqkj_iqpl(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_pqkj_iqpl(3, 5, 2, 7, 13, 11, -1.25) |

    test_ijkl_pqkj_qipl(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_qipl(2, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_qipl(1, 2, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_qipl(1, 1, 2, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_qipl(1, 1, 1, 2, 1, 1, 0.0) |
    test_ijkl_pqkj_qipl(1, 1, 1, 1, 2, 1, 0.0) |
    test_ijkl_pqkj_qipl(1, 1, 1, 1, 1, 2, 0.0) |
    test_ijkl_pqkj_qipl(2, 3, 2, 3, 2, 3, 0.0) |
    test_ijkl_pqkj_qipl(3, 5, 1, 7, 13, 11, 0.0) |
    test_ijkl_pqkj_qipl(3, 5, 2, 7, 13, 11, 0.0) |
    test_ijkl_pqkj_qipl(1, 1, 1, 1, 1, 1, 0.0) |
    test_ijkl_pqkj_qipl(1, 1, 1, 1, 1, 1, -0.5) |
    test_ijkl_pqkj_qipl(2, 1, 1, 1, 1, 1, 2.0) |
    test_ijkl_pqkj_qipl(1, 2, 1, 1, 1, 1, -1.0) |
    test_ijkl_pqkj_qipl(1, 1, 2, 1, 1, 1, 3.7) |
    test_ijkl_pqkj_qipl(1, 1, 1, 2, 1, 1, 1.0) |
    test_ijkl_pqkj_qipl(1, 1, 1, 1, 2, 1, -1.2) |
    test_ijkl_pqkj_qipl(1, 1, 1, 1, 1, 2, 0.7) |
    test_ijkl_pqkj_qipl(2, 3, 2, 3, 2, 3, 12.3) |
    test_ijkl_pqkj_qipl(3, 5, 1, 7, 13, 11, -1.25) |
    test_ijkl_pqkj_qipl(3, 5, 2, 7, 13, 11, -1.25) |

    test_ij_ipqr_jpqr(1, 1, 1, 1, 1) |
    test_ij_ipqr_jpqr(1, 1, 1, 1, 4) |
    test_ij_ipqr_jpqr(1, 1, 1, 4, 1) |
    test_ij_ipqr_jpqr(1, 1, 4, 1, 1) |
    test_ij_ipqr_jpqr(1, 4, 1, 1, 1) |
    test_ij_ipqr_jpqr(4, 1, 1, 1, 1) |
    test_ij_ipqr_jpqr(4, 1, 1, 7, 7) |
    test_ij_ipqr_jpqr(1, 4, 1, 7, 7) |
    test_ij_ipqr_jpqr(3, 4, 5, 6, 7) |

    test_ij_ipqr_jpqr_a(1, 1, 1, 1, 1, 0.5) |
    test_ij_ipqr_jpqr_a(1, 1, 1, 1, 4, -1.0) |
    test_ij_ipqr_jpqr_a(1, 1, 1, 4, 1, 1.5) |
    test_ij_ipqr_jpqr_a(1, 1, 4, 1, 1, -1.2) |
    test_ij_ipqr_jpqr_a(1, 4, 1, 1, 1, 0.8) |
    test_ij_ipqr_jpqr_a(4, 1, 1, 1, 1, 1.4) |
    test_ij_ipqr_jpqr_a(4, 1, 1, 7, 7, 0.4) |
    test_ij_ipqr_jpqr_a(1, 4, 1, 7, 7, -0.6) |
    test_ij_ipqr_jpqr_a(3, 4, 5, 6, 7, -2.0) |

    test_ij_ipqr_pjrq(1, 1, 1, 1, 1, 0.0) |
    test_ij_ipqr_pjrq(4, 1, 1, 7, 7, 0.0) |
    test_ij_ipqr_pjrq(1, 1, 1, 1, 2, 0.0) |
    test_ij_ipqr_pjrq(1, 1, 1, 2, 1, 0.0) |
    test_ij_ipqr_pjrq(1, 1, 2, 1, 1, 0.0) |
    test_ij_ipqr_pjrq(1, 2, 1, 1, 1, 0.0) |
    test_ij_ipqr_pjrq(2, 1, 1, 1, 1, 0.0) |
    test_ij_ipqr_pjrq(3, 3, 3, 3, 3, 0.0) |
    test_ij_ipqr_pjrq(11, 5, 7, 3, 4, 0.0) |
    test_ij_ipqr_pjrq(16, 16, 16, 16, 16, 0.0) |
    test_ij_ipqr_pjrq(1, 1, 1, 1, 1, -0.5) |
    test_ij_ipqr_pjrq(4, 1, 1, 7, 7, 1.2) |
    test_ij_ipqr_pjrq(1, 1, 1, 1, 2, 2.0) |
    test_ij_ipqr_pjrq(1, 1, 1, 2, 1, -1.0) |
    test_ij_ipqr_pjrq(1, 1, 2, 1, 1, 3.7) |
    test_ij_ipqr_pjrq(1, 2, 1, 1, 1, 1.0) |
    test_ij_ipqr_pjrq(2, 1, 1, 1, 1, -0.9) |
    test_ij_ipqr_pjrq(3, 3, 3, 3, 3, 1.0) |
    test_ij_ipqr_pjrq(11, 5, 7, 3, 4, -1.2) |
    test_ij_ipqr_pjrq(16, 16, 16, 16, 16, 0.7) |

    test_ij_jpqr_iprq(1, 1, 1, 1, 1, -1.0) |
    test_ij_jpqr_iprq(1, 1, 1, 1, 3, 0.5) |
    test_ij_jpqr_iprq(1, 1, 1, 3, 1, -1.2) |
    test_ij_jpqr_iprq(1, 1, 3, 1, 1, 0.2) |
    test_ij_jpqr_iprq(1, 3, 1, 1, 1, -0.5) |
    test_ij_jpqr_iprq(3, 1, 1, 1, 1, -0.4) |
    test_ij_jpqr_iprq(3, 3, 3, 2, 4, 1.0) |

    test_ij_pqir_pqjr(3, 4, 5, 6, 7) |
    test_ij_pqir_pqjr_a(3, 4, 5, 6, 7, 2.0) |
    test_ij_pqir_pqjr(3, 3, 3, 3, 3) |
    test_ij_pqir_pqjr(3, 1, 3, 1, 2) |
    test_ij_pqir_pqjr(3, 3, 1, 1, 2) |

    test_ijkl_pi_jklp(1, 4, 5, 6, 2) |
    test_ijkl_pi_jklp(3, 4, 5, 6, 7) |
    test_ijkl_pi_jklp(10, 10, 10, 10, 6) |
    test_ijkl_pi_jklp_a(3, 4, 5, 6, 7, 1.0) |
    test_ijkl_pi_jklp_a(3, 4, 5, 6, 7, 0.0) |
    test_ijkl_pi_jklp_a(3, 4, 5, 6, 7, -2.0) |

    test_jikl_pi_jpkl(1, 4, 5, 6, 2) |
    test_jikl_pi_jpkl(3, 4, 5, 6, 7) |
    test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, 0.0) |
    test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, -2.0) |

    test_ijkl_ijp_klp(1, 1, 1, 1, 1) |
    test_ijkl_ijp_klp(3, 4, 5, 6, 7) |
    test_ijkl_ijp_klp(5, 6, 3, 4, 7) |
    test_ijkl_ijp_klp(1, 100, 1, 100, 100) |
    test_ijkl_ijp_klp_a(3, 4, 5, 6, 7, -2.0) |

    test_ijkl_ij_kl(3, 4, 5, 6) |

    test_ijkl_ij_lk(3, 4, 5, 6) |

    0;


    } catch(...) {
        allocator::shutdown();
        throw;
    }

    allocator::shutdown();

    return rc;
}

