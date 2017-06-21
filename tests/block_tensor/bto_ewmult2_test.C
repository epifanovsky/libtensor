#include <iomanip>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/bto_ewmult2.h>
#include <libtensor/block_tensor/bto_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/dense_tensor/to_ewmult2.h>
#include "bto_ewmult2_test.h"
#include "../compare_ref.h"

namespace libtensor {

void bto_ewmult2_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing bto_ewmult2_test_x<double>    ";
    bto_ewmult2_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing bto_ewmult2_test_x<float>    ";
    bto_ewmult2_test_x<float> t_float;
    t_float.perform();
}

template<>
const double bto_ewmult2_test_x<double>::k_thresh = 1e-15;

template<>
const float bto_ewmult2_test_x<float>::k_thresh = 5e-7;


template<typename T>
void bto_ewmult2_test_x<T>::perform() throw(libtest::test_exception) {

    allocator<T>::init(4, 16, 16777216, 16777216);

    try {

    test_1(false);
    test_1(true);
    test_2(false);
    test_2(true);
    test_3(false);
    test_3(true);
    test_4(false);
    test_4(true);
    test_5(false);
    test_5(true);
    test_6(false);
    test_6(true);
    test_7();

    } catch(...) {
        allocator<T>::shutdown();
        throw;
    }

    allocator<T>::shutdown();
}


/** \test $c_{i} = a_{i} b_{i}$
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_1(bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_1(" << doadd << ")";

    typedef allocator<T> allocator_t;

    try {

    index<1> ia1, ia2;
    index<1> ib1, ib2;
    index<1> ic1, ic2;
    ia2[0] = 9;
    ib2[0] = 9;
    ic2[0] = 9;
    dimensions<1> dimsa(index_range<1>(ia1, ia2));
    dimensions<1> dimsb(index_range<1>(ib1, ib2));
    dimensions<1> dimsc(index_range<1>(ic1, ic2));
    block_index_space<1> bisa(dimsa);
    block_index_space<1> bisb(dimsb);
    block_index_space<1> bisc(dimsc);
    mask<1> m;
    m[0] = true;
    bisa.split(m, 3);
    bisb.split(m, 3);
    bisc.split(m, 3);

    block_tensor<1, T, allocator_t> bta(bisa);
    block_tensor<1, T, allocator_t> btb(bisb);
    block_tensor<1, T, allocator_t> btc(bisc);
    dense_tensor<1, T, allocator_t> ta(dimsa);
    dense_tensor<1, T, allocator_t> tb(dimsb);
    dense_tensor<1, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Fill in random data

    bto_random<1, T>().perform(bta);
    bto_random<1, T>().perform(btb);
    bto_random<1, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<1, T>(bta).perform(ta);
    to_btconv<1, T>(btb).perform(tb);
    to_btconv<1, T>(btc).perform(tc_ref);

    //  Invoke the operation

    T d = drand48();
    bto_ewmult2<0, 0, 1, T> op(bta, btb);
    if(!op.get_bis().equals(bisc)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    if(doadd) {
        to_ewmult2<0, 0, 1, T>(ta, tb, d).perform(false, tc_ref);
        op.perform(btc, d);
    } else {
        to_ewmult2<0, 0, 1, T>(ta, tb).perform(true, tc_ref);
        op.perform(btc);
    }
    to_btconv<1, T>(btc).perform(tc);

    //  Compare against the reference

    compare_ref_x<1, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test $c_{ij} = a_{ij} b_{ij}$
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_2(bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_2(" << doadd << ")";

    typedef allocator<T> allocator_t;

    try {

    index<2> ia1, ia2;
    index<2> ib1, ib2;
    index<2> ic1, ic2;
    ia2[0] = 9; ia2[1] = 5;
    ib2[0] = 9; ib2[1] = 5;
    ic2[0] = 9; ic2[1] = 5;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<2> dimsb(index_range<2>(ib1, ib2));
    dimensions<2> dimsc(index_range<2>(ic1, ic2));
    block_index_space<2> bisa(dimsa);
    block_index_space<2> bisb(dimsb);
    block_index_space<2> bisc(dimsc);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bisa.split(m01, 3);
    bisb.split(m01, 3);
    bisc.split(m01, 3);
    bisa.split(m10, 3);
    bisb.split(m10, 3);
    bisc.split(m10, 3);

    block_tensor<2, T, allocator_t> bta(bisa);
    block_tensor<2, T, allocator_t> btb(bisb);
    block_tensor<2, T, allocator_t> btc(bisc);
    dense_tensor<2, T, allocator_t> ta(dimsa);
    dense_tensor<2, T, allocator_t> tb(dimsb);
    dense_tensor<2, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Fill in random data

    bto_random<2, T>().perform(bta);
    bto_random<2, T>().perform(btb);
    bto_random<2, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<2, T>(bta).perform(ta);
    to_btconv<2, T>(btb).perform(tb);
    to_btconv<2, T>(btc).perform(tc_ref);

    //  Invoke the operation

    T d = drand48();
    permutation<2> perma;
    permutation<2> permb;
    permutation<2> permc;
    bto_ewmult2<0, 0, 2, T> op(bta, perma, btb, permb, permc);
    if(!op.get_bis().equals(bisc)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    if(doadd) {
        to_ewmult2<0, 0, 2, T>(ta, perma, tb, permb, permc, d).
            perform(false, tc_ref);
        op.perform(btc, d);
    } else {
        to_ewmult2<0, 0, 2, T>(ta, perma, tb, permb, permc).
            perform(true, tc_ref);
        op.perform(btc);
    }
    to_btconv<2, T>(btc).perform(tc);

    //  Compare against the reference

    compare_ref_x<2, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test $c_{ij} = a_{ij} b_{ji}$
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_3(bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_3(" << doadd << ")";

    typedef allocator<T> allocator_t;

    try {

    index<2> ia1, ia2;
    index<2> ib1, ib2;
    index<2> ic1, ic2;
    ia2[0] = 9; ia2[1] = 5;
    ib2[0] = 5; ib2[1] = 9;
    ic2[0] = 9; ic2[1] = 5;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<2> dimsb(index_range<2>(ib1, ib2));
    dimensions<2> dimsc(index_range<2>(ic1, ic2));
    block_index_space<2> bisa(dimsa);
    block_index_space<2> bisb(dimsb);
    block_index_space<2> bisc(dimsc);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bisa.split(m01, 3);
    bisb.split(m01, 3);
    bisc.split(m01, 3);
    bisa.split(m10, 3);
    bisb.split(m10, 3);
    bisc.split(m10, 3);

    block_tensor<2, T, allocator_t> bta(bisa);
    block_tensor<2, T, allocator_t> btb(bisb);
    block_tensor<2, T, allocator_t> btc(bisc);
    dense_tensor<2, T, allocator_t> ta(dimsa);
    dense_tensor<2, T, allocator_t> tb(dimsb);
    dense_tensor<2, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Fill in random data

    bto_random<2, T>().perform(bta);
    bto_random<2, T>().perform(btb);
    bto_random<2, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<2, T>(bta).perform(ta);
    to_btconv<2, T>(btb).perform(tb);
    to_btconv<2, T>(btc).perform(tc_ref);

    //  Invoke the operation

    T d = drand48();
    permutation<2> perma;
    permutation<2> permb; permb.permute(0, 1);
    permutation<2> permc;
    bto_ewmult2<0, 0, 2, T> op(bta, perma, btb, permb, permc);
    if(!op.get_bis().equals(bisc)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    if(doadd) {
        to_ewmult2<0, 0, 2, T>(ta, perma, tb, permb, permc, d).
            perform(false, tc_ref);
        op.perform(btc, d);
    } else {
        to_ewmult2<0, 0, 2, T>(ta, perma, tb, permb, permc).
            perform(true, tc_ref);
        op.perform(btc);
    }
    to_btconv<2, T>(btc).perform(tc);

    //  Compare against the reference

    compare_ref_x<2, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test $c_{ijkl} = a_{kj} b_{ikl}$
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_4(bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_4(" << doadd << ")";

    typedef allocator<T> allocator_t;

    try {

    index<2> ia1, ia2;
    index<3> ib1, ib2;
    index<4> ic1, ic2;
    ia2[0] = 9; ia2[1] = 5;
    ib2[0] = 5; ib2[1] = 9; ib2[2] = 9;
    ic2[0] = 5; ic2[1] = 5; ic2[2] = 9; ic2[3] = 9;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dimsa);
    block_index_space<3> bisb(dimsb);
    block_index_space<4> bisc(dimsc);
    mask<2> m01, m10;
    mask<3> m011, m100;
    mask<4> m0011, m0100, m1000;
    m10[0] = true; m01[1] = true;
    m100[0] = true; m011[1] = true; m011[2] = true;
    m1000[0] = true; m0100[1] = true; m0011[2] = true; m0011[3] = true;
    bisa.split(m01, 3);
    bisa.split(m10, 3);
    bisa.split(m10, 5);
    bisb.split(m011, 3);
    bisb.split(m011, 5);
    bisb.split(m100, 3);
    bisc.split(m0011, 3);
    bisc.split(m0011, 5);
    bisc.split(m0100, 3);
    bisc.split(m1000, 3);

    block_tensor<2, T, allocator_t> bta(bisa);
    block_tensor<3, T, allocator_t> btb(bisb);
    block_tensor<4, T, allocator_t> btc(bisc);
    dense_tensor<2, T, allocator_t> ta(dimsa);
    dense_tensor<3, T, allocator_t> tb(dimsb);
    dense_tensor<4, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Fill in random data

    bto_random<2, T>().perform(bta);
    bto_random<3, T>().perform(btb);
    bto_random<4, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<2, T>(bta).perform(ta);
    to_btconv<3, T>(btb).perform(tb);
    to_btconv<4, T>(btc).perform(tc_ref);

    //  Invoke the operation

    T d = drand48();
    permutation<2> perma; perma.permute(0, 1); // kj->jk
    permutation<3> permb; permb.permute(1, 2); // ikl->ilk
    permutation<4> permc; permc.permute(0, 1).permute(2, 3); // jilk->ijkl
    bto_ewmult2<1, 2, 1, T> op(bta, perma, btb, permb, permc);
    if(!op.get_bis().equals(bisc)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    if(doadd) {
        to_ewmult2<1, 2, 1, T>(ta, perma, tb, permb, permc, d).
            perform(false, tc_ref);
        op.perform(btc, d);
    } else {
        to_ewmult2<1, 2, 1, T>(ta, perma, tb, permb, permc).
            perform(true, tc_ref);
        op.perform(btc);
    }
    to_btconv<4, T>(btc).perform(tc);

    //  Compare against the reference

    compare_ref_x<4, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test $c_{ijkl} = a_{ljk} b_{jil}$
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_5(bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_5(" << doadd << ")";

    typedef allocator<T> allocator_t;

    try {

    index<3> ia1, ia2;
    index<3> ib1, ib2;
    index<4> ic1, ic2;
    ia2[0] = 9; ia2[1] = 9; ia2[2] = 5;
    ib2[0] = 9; ib2[1] = 5; ib2[2] = 9;
    ic2[0] = 5; ic2[1] = 9; ic2[2] = 5; ic2[3] = 9;
    dimensions<3> dimsa(index_range<3>(ia1, ia2));
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    block_index_space<3> bisa(dimsa);
    block_index_space<3> bisb(dimsb);
    block_index_space<4> bisc(dimsc), bisc_ref(dimsc);
    mask<3> m001, m010, m101, m110;
    mask<4> m0010, m0101, m1000, m1010;
    m110[0] = true; m110[1] = true; m001[2] = true;
    m101[0] = true; m010[1] = true; m101[2] = true;
    m1000[0] = true; m0010[2] = true;
    m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
    bisa.split(m001, 3);
    bisa.split(m110, 3);
    bisa.split(m110, 5);
    bisb.split(m010, 3);
    bisb.split(m101, 3);
    bisb.split(m101, 5);
    bisc.split(m0101, 3);
    bisc.split(m0101, 5);
    bisc.split(m1010, 3);
    bisc_ref.split(m0101, 3);
    bisc_ref.split(m0101, 5);
    bisc_ref.split(m1010, 3);
    bisc_ref.split(m1000, 3);

    block_tensor<3, T, allocator_t> bta(bisa);
    block_tensor<3, T, allocator_t> btb(bisb);
    block_tensor<4, T, allocator_t> btc(bisc);
    dense_tensor<3, T, allocator_t> ta(dimsa);
    dense_tensor<3, T, allocator_t> tb(dimsb);
    dense_tensor<4, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Fill in random data

    bto_random<3, T>().perform(bta);
    bto_random<3, T>().perform(btb);
    bto_random<4, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<3, T>(bta).perform(ta);
    to_btconv<3, T>(btb).perform(tb);
    to_btconv<4, T>(btc).perform(tc_ref);

    //  Invoke the operation

    T d = drand48();
    permutation<3> perma; perma.permute(1, 2).permute(0, 1); // ljk->klj
    permutation<3> permb; permb.permute(0, 1).permute(1, 2); // jil->ilj
    permutation<4> permc;
    permc.permute(2, 3).permute(1, 2).permute(0, 2); // kilj->ijkl
    bto_ewmult2<1, 1, 2, T> op(bta, perma, btb, permb, permc);
    if(!op.get_bis().equals(bisc_ref)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    if(doadd) {
        to_ewmult2<1, 1, 2, T>(ta, perma, tb, permb, permc, d).
            perform(false, tc_ref);
        op.perform(btc, d);
    } else {
        to_ewmult2<1, 1, 2, T>(ta, perma, tb, permb, permc).
            perform(true, tc_ref);
        op.perform(btc);
    }
    to_btconv<4, T>(btc).perform(tc);

    //  Compare against the reference

    compare_ref_x<4, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test $c_{ijkl} = a_{ikl} b_{jkl}$, perm symmetry
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_6(bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_6(" << doadd << ")";

    typedef allocator<T> allocator_t;

    try {

    size_t ni = 10, nj = 10, nk = 8, nl = 8;
    index<3> ia1, ia2;
    index<3> ib1, ib2;
    index<4> ic1, ic2;
    ia2[0] = ni - 1; ia2[1] = nk - 1; ia2[2] = nl - 1;
    ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = nl - 1;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
    dimensions<3> dimsa(index_range<3>(ia1, ia2));
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    block_index_space<3> bisa(dimsa);
    block_index_space<3> bisb(dimsb);
    block_index_space<4> bisc(dimsc), bisc_ref(dimsc);
    mask<3> m011, m100;
    mask<4> m0011, m0100, m1000, m1100;
    m100[0] = true; m011[1] = true; m011[2] = true;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
    m1000[0] = true; m0100[1] = true;
    bisa.split(m011, 3);
    bisa.split(m100, 3);
    bisa.split(m100, 6);
    bisb.split(m011, 3);
    bisb.split(m100, 3);
    bisb.split(m100, 6);
    bisc.split(m0011, 3);
    bisc.split(m1100, 3);
    bisc.split(m1100, 6);
    bisc_ref.split(m0011, 3);
    bisc_ref.split(m1000, 3);
    bisc_ref.split(m1000, 6);
    bisc_ref.split(m0100, 3);
    bisc_ref.split(m0100, 6);

    block_tensor<3, T, allocator_t> bta(bisa);
    block_tensor<3, T, allocator_t> btb(bisb);
    block_tensor<4, T, allocator_t> btc(bisc);
    symmetry<4, T> symc(bisc), symc_ref(bisc);
    dense_tensor<3, T, allocator_t> ta(dimsa);
    dense_tensor<3, T, allocator_t> tb(dimsb);
    dense_tensor<4, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Set up symmetry

    scalar_transf<T> tr0, tr1(-1.);
    {
        block_tensor_ctrl<3, T> ca(bta);
        se_perm<3, T> seperm(permutation<3>().permute(1, 2), tr0);
        ca.req_symmetry().insert(seperm);
    }
    {
        block_tensor_ctrl<3, T> cb(btb);
        se_perm<3, T> seperm(permutation<3>().permute(1, 2), tr0);
        cb.req_symmetry().insert(seperm);
    }
    {
        se_perm<4, T> seperm(permutation<4>().permute(2, 3), tr0);
        symc_ref.insert(seperm);
        if (doadd) {
            block_tensor_ctrl<4, T> cc(btc);
            cc.req_symmetry().insert(seperm);
        }
    }

    //  Fill in random data

    bto_random<3, T>().perform(bta);
    bto_random<3, T>().perform(btb);
    bto_random<4, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<3, T>(bta).perform(ta);
    to_btconv<3, T>(btb).perform(tb);
    to_btconv<4, T>(btc).perform(tc_ref);

    //  Invoke the operation

    T d = drand48();
    permutation<3> perma;
    permutation<3> permb;
    permutation<4> permc;
    bto_ewmult2<1, 1, 2, T> op(bta, perma, btb, permb, permc);
    if(!op.get_bis().equals(bisc_ref)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    if(doadd) {
        to_ewmult2<1, 1, 2, T>(ta, perma, tb, permb, permc, d).
            perform(false, tc_ref);
        op.perform(btc, d);
    } else {
        to_ewmult2<1, 1, 2, T>(ta, perma, tb, permb, permc).
            perform(true, tc_ref);
        op.perform(btc);
    }
    to_btconv<4, T>(btc).perform(tc);

    //  Compare against the reference

    {
        block_tensor_ctrl<4, T> cc(btc);
        so_copy<4, T>(cc.req_const_symmetry()).perform(symc);
    }

    compare_ref_x<4, T>::compare(tnss.str().c_str(), symc, symc_ref);
    compare_ref_x<4, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test $c_{ijk} = a_{ij} b_{ik}$, label symmetry
 **/
template<typename T>
void bto_ewmult2_test_x<T>::test_7() throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "bto_ewmult2_test_x<T>::test_7()";

    typedef allocator<T> allocator_t;

    static const char *pgsym = "cs";
    std::vector<std::string> irnames(2);
    irnames[0] = "A"; irnames[1] = "B";
    point_group_table pgt(pgsym, irnames, irnames[0]);
    pgt.add_product(0, 0, 0);
    pgt.add_product(0, 1, 1);
    pgt.add_product(1, 1, 0);
    product_table_container::get_instance().add(pgt);

    try {

    size_t ni = 10, nj = 9, nk = 8;
    index<2> ia1, ia2;
    index<2> ib1, ib2;
    index<3> ic1, ic2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    ib2[0] = ni - 1; ib2[1] = nk - 1;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<2> dimsb(index_range<2>(ib1, ib2));
    dimensions<3> dimsc(index_range<3>(ic1, ic2));
    block_index_space<2> bisa(dimsa);
    block_index_space<2> bisb(dimsb);
    block_index_space<3> bisc(dimsc), bisc_ref(dimsc);
    mask<2> m01, m10;
    mask<3> m001, m010, m100;
    m10[0] = true; m01[1] = true;
    m100[0] = true; m010[1] = true; m001[2] = true;
    bisa.split(m10, 3);
    bisa.split(m10, 5);
    bisa.split(m10, 8);
    bisa.split(m01, 6);
    bisb.split(m10, 3);
    bisb.split(m10, 5);
    bisb.split(m10, 8);
    bisb.split(m01, 6);
    bisc.split(m100, 3);
    bisc.split(m100, 5);
    bisc.split(m100, 8);
    bisc.split(m010, 6);
    bisc.split(m001, 6);
    bisc_ref.split(m100, 3);
    bisc_ref.split(m100, 5);
    bisc_ref.split(m100, 8);
    bisc_ref.split(m010, 6);
    bisc_ref.split(m001, 6);

    block_tensor<2, T, allocator_t> bta(bisa);
    block_tensor<2, T, allocator_t> btb(bisb);
    block_tensor<3, T, allocator_t> btc(bisc);
    symmetry<3, T> symc(bisc), symc_ref(bisc);
    dense_tensor<2, T, allocator_t> ta(dimsa);
    dense_tensor<2, T, allocator_t> tb(dimsb);
    dense_tensor<3, T, allocator_t> tc(dimsc), tc_ref(dimsc);

    //  Set up symmetry

    {
        block_tensor_ctrl<2, T> ca(bta);
        se_label<2, T> selabel(bisa.get_block_index_dims(), pgsym);
        block_labeling<2> &bl = selabel.get_labeling();
        bl.assign(m10, 0, 0);
        bl.assign(m10, 1, 1);
        bl.assign(m10, 2, 0);
        bl.assign(m10, 3, 1);
        bl.assign(m01, 0, 0);
        bl.assign(m01, 1, 1);
        product_table_i::label_set_t ls;
        ls.insert(0); ls.insert(1);
        selabel.set_rule(ls);
        ca.req_symmetry().insert(selabel);
    }
    {
        block_tensor_ctrl<2, T> cb(btb);
        se_label<2, T> selabel(bisb.get_block_index_dims(), pgsym);
        block_labeling<2> &bl = selabel.get_labeling();
        bl.assign(m10, 0, 0);
        bl.assign(m10, 1, 1);
        bl.assign(m10, 2, 0);
        bl.assign(m10, 3, 1);
        bl.assign(m01, 0, 0);
        bl.assign(m01, 1, 1);
        product_table_i::label_set_t ls;
        ls.insert(0); ls.insert(1);
        selabel.set_rule(ls);
        cb.req_symmetry().insert(selabel);
    }
    {
        se_label<3, T> selabel(bisc_ref.get_block_index_dims(),
            pgsym);
        block_labeling<3> &bl = selabel.get_labeling();
        bl.assign(m100, 0, 0);
        bl.assign(m100, 1, 1);
        bl.assign(m100, 2, 0);
        bl.assign(m100, 3, 1);
        bl.assign(m010, 0, 0);
        bl.assign(m010, 1, 1);
        bl.assign(m001, 0, 0);
        bl.assign(m001, 1, 1);
        product_table_i::label_set_t ls;
        ls.insert(0); ls.insert(1);
        selabel.set_rule(ls);
        symc_ref.insert(selabel);
    }

    //  Fill in random data

    bto_random<2, T>().perform(bta);
    bto_random<2, T>().perform(btb);
    bto_random<3, T>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    to_btconv<2, T>(bta).perform(ta);
    to_btconv<2, T>(btb).perform(tb);
    to_btconv<3, T>(btc).perform(tc_ref);

    //  Invoke the operation

    permutation<2> perma; perma.permute(0, 1);
    permutation<2> permb; permb.permute(0, 1);
    permutation<3> permc; permc.permute(1, 2).permute(0, 1);
    bto_ewmult2<1, 1, 1, T> op(bta, perma, btb, permb, permc);
    if(!op.get_bis().equals(bisc_ref)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
    }
    to_ewmult2<1, 1, 1, T>(ta, perma, tb, permb, permc).
        perform(true, tc_ref);
    op.perform(btc);
    to_btconv<3, T>(btc).perform(tc);

    //  Compare against the reference

    {
        block_tensor_ctrl<3, T> cc(btc);
        so_copy<3, T>(cc.req_const_symmetry()).perform(symc);
    }

    compare_ref_x<3, T>::compare(tnss.str().c_str(), symc, symc_ref);
    compare_ref_x<3, T>::compare(tnss.str().c_str(), tc, tc_ref, k_thresh);

    } catch(exception &e) {
        product_table_container::get_instance().erase(pgsym);
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        product_table_container::get_instance().erase(pgsym);
        throw;
    }

    product_table_container::get_instance().erase(pgsym);
}


} // namespace libtensor
