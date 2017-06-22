#include <libtensor/core/allocator.h>
#include <iostream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/dense_tensor/to_contract2.h>
#include <libtensor/libtensor.h>
#include "../compare_ref.h"
#include "expr_test_float_double.h"

namespace libtensor {

void expr_test_new::perform() throw(libtest::test_exception) {
    std::cout << "Testing expr_test_x<double>   ";
    expr_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing expr_test_x<float>   ";
    expr_test_x<float> t_float;
    t_float.perform();

}

template<>
const double expr_test_x<double>::k_thresh = 5e-15;

template<>
const float expr_test_x<float>::k_thresh = 1e-6;

template<typename T>
void expr_test_x<T>::perform() throw(libtest::test_exception) {

    allocator<T>::init(16, 16, 16777216, 16777216);

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();
        test_6();
        test_7();
        test_8();
        test_9();
        test_10();
        test_11();
        test_12();
        test_13();
        test_14();

    } catch(...) {
        allocator<T>::shutdown();
        throw;
    }

    allocator<T>::shutdown();
}


template<typename T>
void expr_test_x<T>::test_1() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_1()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2, T> t1(sov), t2(sov), t3(sov), t3_ref(sov);

    bto_random<2, T>().perform(t1);
    bto_random<2, T>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    bto_add<2, T> op(t1);
    op.add_op(t2, -1.0);
    op.perform(t3_ref);

    letter i, a;

    t3(i|a) = t1(i|a) - t2(i|a);

    compare_ref_x<2, T>::compare(testname, t3, t3_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_2() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_2()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<2, T> t1(sov);
    btensor<4, T> t2(soovv);
    btensor<2, T> f_ov(sov);
    btensor<4, T> i_ooov(sooov), i_oovv(soovv), i_ovov(sovov), i_ovvv(sovvv);
    btensor<4, T> i3_ovvv(sovvv), i5_vvvv(svvvv);

    bto_random<2, T>().perform(t1);
    bto_random<4, T>().perform(t2);
    bto_random<2, T>().perform(f_ov);
    bto_random<4, T>().perform(i_ooov);
    bto_random<4, T>().perform(i_oovv);
    bto_random<4, T>().perform(i_ovov);
    bto_random<4, T>().perform(i_ovvv);
    bto_random<4, T>().perform(i5_vvvv);

    letter i, j, k, a, b, c, d;

    i3_ovvv(i|a|b|c) =
          i_ovvv(i|a|b|c)
        + asymm(b, c, contract(j,
            t1(j|c),
            i_ovov(j|b|i|a)
            - contract(k|d, t2(i|k|b|d), i_oovv(j|k|a|d))))
        - asymm(b, c, contract(k|d, i_ovvv(k|c|a|d), t2(i|k|b|d)));

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_3() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_3()";

    try {

    bispace<1> so(10); so.split(5);
    bispace<1> sv(4); sv.split(2);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<2, T> t1(sov);
    btensor<4, T> t2(soovv);
    btensor<4, T> i1_ovov(sovov);

    {
        block_tensor_ctrl<4, T> c_t2(t2);
        symmetry<4, T> sym_t2(t2.get_bis());
        scalar_transf<T> tr1(-1.);
        sym_t2.insert(se_perm<4, T>(permutation<4>().
            permute(0, 1), tr1));
        sym_t2.insert(se_perm<4, T>(permutation<4>().
            permute(2, 3), tr1));
        so_copy<4, T>(sym_t2).perform(c_t2.req_symmetry());
    }

    bto_random<2, T>().perform(t1);
    bto_random<4, T>().perform(t2);
    bto_random<4, T>().perform(i1_ovov);

    letter i, k, b, c;

    i1_ovov(i|b|k|c) = t2(i|k|b|c) - t1(i|c) * t1(k|b);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_4() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_4()";

    try {

    bispace<1> so(10); so.split(5);
    bispace<1> sv(4); sv.split(2);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<4, T> i_oovv(soovv), i_ovov(sovov), t2(soovv);
    btensor<4, T> i2_oovv(soovv);

    {
        block_tensor_ctrl<4, T> c_i_oovv(i_oovv), c_t2(t2);
        scalar_transf<T> tr1(-1.);
        symmetry<4, T> sym_t2(t2.get_bis());
        sym_t2.insert(se_perm<4, T>(permutation<4>().
            permute(0, 1), tr1));
        sym_t2.insert(se_perm<4, T>(permutation<4>().
            permute(2, 3), tr1));
        so_copy<4, T>(sym_t2).perform(c_t2.req_symmetry());
        so_copy<4, T>(sym_t2).perform(c_i_oovv.req_symmetry());
    }
    {
        scalar_transf<T> tr0;
        block_tensor_ctrl<4, T> c_i_ovov(i_ovov);
        symmetry<4, T> sym_i_ovov(i_ovov.get_bis());
        sym_i_ovov.insert(se_perm<4, T>(permutation<4>().
            permute(0, 2).permute(1, 3), tr0));
        so_copy<4, T>(sym_i_ovov).perform(c_i_ovov.req_symmetry());
    }

    bto_random<4, T>().perform(i_oovv);
    bto_random<4, T>().perform(i_ovov);
    bto_random<4, T>().perform(t2);

    letter j, k, l, a, b, c;

    i2_oovv(j|k|a|b) =
        i_ovov(k|a|j|b) - contract(l|c, i_oovv(k|l|b|c), t2(j|l|a|c));

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_5() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_5()";

    try {

    bispace<1> so(10); so.split(5);
    bispace<1> sv(4); sv.split(2);

    bispace<2> soo(so&so);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    btensor<4, T> t1_oovv(soovv), t2_oovv(soovv);
    btensor<2, T> t3_oo(soo), t3_oo_ref(soo);

    {
        scalar_transf<T> tr1(-1.);
        block_tensor_ctrl<4, T> c_t1_oovv(t1_oovv);
        c_t1_oovv.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(0, 1), tr1));
        c_t1_oovv.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(2, 3), tr1));
    }

    bto_random<4, T>().perform(t1_oovv);
    bto_random<4, T>().perform(t2_oovv);

    letter i, j, k, a, b;

    t3_oo(i|j) = contract(k|a|b, t1_oovv(j|k|a|b), t2_oovv(i|k|a|b));

    contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
    contr.contract(1, 1);
    contr.contract(2, 2);
    contr.contract(3, 3);
    bto_contract2<1, 1, 3, T>(contr, t1_oovv, t2_oovv).perform(t3_oo_ref);
    compare_ref_x<2, T>::compare(testname, t3_oo, t3_oo_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_6() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_6()";

    typedef allocator<T> allocator_t;

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(4); sv.split(1).split(2).split(3);

    bispace<2> sov(so|sv);
    bispace<4> soooo(so&so&so&so), sooov((so&so&so)|sv), soovv((so&so)|(sv&sv));

    btensor<4, T> i_oooo(soooo), i_ooov(sooov), i_oovv(soovv);
    btensor<2, T> t1(sov);
    btensor<4, T> t2(soovv);
    btensor<4, T> i4_oooo(soooo), i4_oooo_ref(soooo);

    {
        scalar_transf<T> tr1(-1.);
        block_tensor_ctrl<4, T> c_i_oooo(i_oooo);
        c_i_oooo.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(0, 1), tr1));
        c_i_oooo.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(2, 3), tr1));

        block_tensor_ctrl<4, T> c_i_ooov(i_ooov);
        c_i_ooov.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(0, 1), tr1));

        block_tensor_ctrl<4, T> c_i_oovv(i_oovv);
        c_i_oovv.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(0, 1), tr1));
        c_i_oovv.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(2, 3), tr1));

        block_tensor_ctrl<4, T> c_t2(t2);
        c_t2.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(0, 1), tr1));
        c_t2.req_symmetry().insert(se_perm<4, T>(
            permutation<4>().permute(2, 3), tr1));
    }

    bto_random<4, T>().perform(i_oooo);
    bto_random<4, T>().perform(i_ooov);
    bto_random<4, T>().perform(i_oovv);
    bto_random<2, T>().perform(t1);
    bto_random<4, T>().perform(t2);

    letter i, j, k, l, a, b;

    T multiplier = 0.5;

    i4_oooo(i|j|k|l) =
          i_oooo(i|j|k|l)
        + multiplier * contract(a|b, i_oovv(k|l|a|b), t2(i|j|a|b))
        + asymm(i, j, contract(a, i_ooov(k|l|i|a), t1(j|a)))
        + contract(a|b, i_oovv(k|l|a|b), t1(i|a)*t1(j|b));

    bto_copy<4, T> op1(i_oooo);

    contraction2<2, 2, 2> contr2;
    contr2.contract(2, 2);
    contr2.contract(3, 3);
    bto_contract2<2, 2, 2, T> op2(contr2, t2, i_oovv);

    contraction2<3, 1, 1> contr3(permutation<4>().
        permute(0, 2).permute(1, 3));
    contr3.contract(3, 1);
    bto_contract2<3, 1, 1, T> op3a(contr3, i_ooov, t1);
    bto_symmetrize2<4, T> op3(op3a, 0, 1, false);

    contraction2<2, 2, 0> contr4a(permutation<4>().permute(1, 2));
    btensor<4, T> tmp4a(soovv);
    bto_contract2<2, 2, 0, T>(contr4a, t1, t1).perform(tmp4a);
    contraction2<2, 2, 2> contr4;
    contr4.contract(2, 2);
    contr4.contract(3, 3);
    bto_contract2<2, 2, 2, T> op4(contr4, tmp4a, i_oovv);

    op1.perform(i4_oooo_ref);
    op2.perform(i4_oooo_ref, 0.5);
    op3.perform(i4_oooo_ref, 1.0);
    op4.perform(i4_oooo_ref, 1.0);

    compare_ref_x<4, T>::compare(testname, i4_oooo, i4_oooo_ref, k_thresh);

    dense_tensor<4, T, allocator_t> ti_oooo(i_oooo.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> ti_ooov(i_ooov.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> ti_oovv(i_oovv.get_bis().get_dims());
    dense_tensor<2, T, allocator_t> tt1(t1.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> tt2(t2.get_bis().get_dims());
    to_btconv<4, T>(i_oooo).perform(ti_oooo);
    to_btconv<4, T>(i_ooov).perform(ti_ooov);
    to_btconv<4, T>(i_oovv).perform(ti_oovv);
    to_btconv<2, T>(t1).perform(tt1);
    to_btconv<4, T>(t2).perform(tt2);

    to_copy<4, T> top1(ti_oooo);

    to_contract2<2, 2, 2, T> top2(contr2, tt2, ti_oovv, 0.5);

    dense_tensor<4, T, allocator_t> ttmp3a(i_oooo.get_bis().get_dims());
    to_contract2<3, 1, 1, T>(contr3, ti_ooov, tt1).perform(true, ttmp3a);
    to_add<4, T> top3(ttmp3a);
    top3.add_op(ttmp3a, permutation<4>().permute(0, 1), -1.0);

    dense_tensor<4, T, allocator_t> ttmp4a(i_oovv.get_bis().get_dims());
    to_contract2<2, 2, 0, T>(contr4a, tt1, tt1).perform(true, ttmp4a);
    to_contract2<2, 2, 2, T> top4(contr4, ttmp4a, ti_oovv);

    dense_tensor<4, T, allocator_t> ti4_oooo(i4_oooo.get_bis().get_dims()),
        ti4_oooo_ref(i4_oooo.get_bis().get_dims());
    top1.perform(true, ti4_oooo_ref);
    top2.perform(false, ti4_oooo_ref);
    top3.perform(false, ti4_oooo_ref);
    top4.perform(false, ti4_oooo_ref);
    to_btconv<4, T>(i4_oooo).perform(ti4_oooo);

    compare_ref_x<4, T>::compare(testname, ti4_oooo, ti4_oooo_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_7() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_7()";

    typedef allocator<T> allocator_t;

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(4); sv.split(1).split(2).split(3);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)), sovvv(so|(sv&sv&sv));

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<4, T> i_ooov(sooov), i_oovv(soovv), i_ovov(sovov), i_ovvv(sovvv);
    btensor<2, T> t1(sov);
    btensor<4, T> t2(soovv);
    btensor<4, T> i1_ovov(sovov), i1_ovov_ref(sovov);

    bto_random<4, T>().perform(i_ooov);
    bto_random<4, T>().perform(i_oovv);
    bto_random<4, T>().perform(i_ovov);
    bto_random<4, T>().perform(i_ovvv);
    bto_random<2, T>().perform(t1);
    bto_random<4, T>().perform(t2);

    letter i, j, k, a, b, c;

    T multiplier1 = 0.5, multiplier2 = 2.0;

    i1_ovov(i|a|j|b) =
          i_ovov(i|a|j|b)
        - contract(c, i_ovvv(i|a|b|c), t1(j|c))
        - contract(k, i_ooov(i|k|j|b), t1(k|a))
        + multiplier1 * contract(k|c,
                          t2(j|k|c|a) + multiplier2 * t1(j|c)*t1(k|a),
                               i_oovv(i|k|b|c));

    bto_copy<4, T> op1(i_ovov);

    contraction2<3, 1, 1> contr2(permutation<4>().permute(2, 3));
    contr2.contract(3, 1);
    bto_contract2<3, 1, 1, T> op2(contr2, i_ovvv, t1);

    contraction2<3, 1, 1> contr3(permutation<4>().
        permute(1, 3).permute(2, 3));
    contr3.contract(1, 0);
    bto_contract2<3, 1, 1, T> op3(contr3, i_ooov, t1);

    contraction2<2, 2, 0> contr4a(permutation<4>().permute(1, 2));
    btensor<4, T> tmp4a(soovv);
    bto_copy<4, T>(t2).perform(tmp4a);
    bto_contract2<2, 2, 0, T>(contr4a, t1, t1).perform(tmp4a, 2.0);

    contraction2<2, 2, 2> contr4(permutation<4>().permute(0, 2));
    contr4.contract(1, 1);
    contr4.contract(2, 3);
    bto_contract2<2, 2, 2, T> op4(contr4, tmp4a, i_oovv);

    op1.perform(i1_ovov_ref);
    op2.perform(i1_ovov_ref, -1.0);
    op3.perform(i1_ovov_ref, -1.0);
    op4.perform(i1_ovov_ref, 0.5);

    compare_ref_x<4, T>::compare(testname, i1_ovov, i1_ovov_ref, k_thresh);

    dense_tensor<4, T, allocator_t> ti_ooov(i_ooov.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> ti_oovv(i_oovv.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> ti_ovov(i_ovov.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> ti_ovvv(i_ovvv.get_bis().get_dims());
    dense_tensor<2, T, allocator_t> tt1(t1.get_bis().get_dims());
    dense_tensor<4, T, allocator_t> tt2(t2.get_bis().get_dims());
    to_btconv<4, T>(i_ooov).perform(ti_ooov);
    to_btconv<4, T>(i_oovv).perform(ti_oovv);
    to_btconv<4, T>(i_ovov).perform(ti_ovov);
    to_btconv<4, T>(i_ovvv).perform(ti_ovvv);
    to_btconv<2, T>(t1).perform(tt1);
    to_btconv<4, T>(t2).perform(tt2);

    to_copy<4, T> top1(ti_ovov);

    to_contract2<3, 1, 1, T> top2(contr2, ti_ovvv, tt1, -1.0);

    to_contract2<3, 1, 1, T> top3(contr3, ti_ooov, tt1, -1.0);

    dense_tensor<4, T, allocator_t> ttmp4a(i_oovv.get_bis().get_dims());
    to_copy<4, T>(tt2).perform(true, ttmp4a);
    to_contract2<2, 2, 0, T>(contr4a, tt1, tt1, 2.0).perform(false, ttmp4a);
    to_contract2<2, 2, 2, T> top4(contr4, ttmp4a, ti_oovv, 0.5);

    dense_tensor<4, T, allocator_t> ti1_ovov(i1_ovov.get_bis().get_dims()),
        ti1_ovov_ref(i1_ovov.get_bis().get_dims());
    top1.perform(true, ti1_ovov_ref);
    top2.perform(false, ti1_ovov_ref);
    top3.perform(false, ti1_ovov_ref);
    top4.perform(false, ti1_ovov_ref);
    to_btconv<4, T>(i1_ovov).perform(ti1_ovov);

    compare_ref_x<4, T>::compare(testname, ti1_ovov, ti1_ovov_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_8() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_8()";

    bool need_erase = true;
    const char *pgtid = "point_group_cs";

    try {

    point_group_table::label_t ap = 0, app = 1;
    std::vector<std::string> irnames(2);
    irnames[ap] = "A'"; irnames[app] = "A''";

    point_group_table cs(pgtid, irnames, irnames[ap]);
    cs.add_product(app, app, ap);
    cs.check();
    product_table_container::get_instance().add(cs);

    {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(8); sv.split(2).split(4).split(6);

    bispace<2> soo(so&so);
    bispace<4> soovv((so&so)|(sv&sv));

    btensor<4, T> i_oovv(soovv), t2(soovv);
    btensor<2, T> f_oo(soo), f2_oo(soo);

    mask<2> m11;
    m11[0] = true; m11[1] = true;
    mask<4> m0011, m1100;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;

    se_label<2, T> l_oo(soo.get_bis().get_block_index_dims(), pgtid);
    block_labeling<2> &bl_oo = l_oo.get_labeling();
    bl_oo.assign(m11, 0, ap);
    bl_oo.assign(m11, 1, app);
    bl_oo.assign(m11, 2, ap);
    bl_oo.assign(m11, 3, app);
    l_oo.set_rule(ap);
    se_label<4, T> l_oovv(soovv.get_bis().get_block_index_dims(),
        pgtid);
    block_labeling<4> &bl_oovv = l_oovv.get_labeling();
    bl_oovv.assign(m1100, 0, ap);
    bl_oovv.assign(m1100, 1, app);
    bl_oovv.assign(m1100, 2, ap);
    bl_oovv.assign(m1100, 3, app);
    bl_oovv.assign(m0011, 0, ap);
    bl_oovv.assign(m0011, 1, app);
    bl_oovv.assign(m0011, 2, ap);
    bl_oovv.assign(m0011, 3, app);
    l_oovv.set_rule(ap);

    {
        block_tensor_ctrl<2, T> c_f_oo(f_oo);
        symmetry<2, T> sym_f_oo(f_oo.get_bis());
        scalar_transf<T> tr0, tr1(-1.);
        sym_f_oo.insert(se_perm<2, T>(permutation<2>().
            permute(0, 1), tr0));
        sym_f_oo.insert(l_oo);
        so_copy<2, T>(sym_f_oo).perform(c_f_oo.req_symmetry());

        block_tensor_ctrl<4, T> c_i_oovv(i_oovv), c_t2(t2);
        symmetry<4, T> sym_t2(t2.get_bis());
        sym_t2.insert(se_perm<4, T>(permutation<4>().
            permute(0, 1), tr1));
        sym_t2.insert(se_perm<4, T>(permutation<4>().
            permute(2, 3), tr1));
        sym_t2.insert(l_oovv);
        so_copy<4, T>(sym_t2).perform(c_t2.req_symmetry());
        so_copy<4, T>(sym_t2).perform(c_i_oovv.req_symmetry());
    }

    bto_random<2, T>().perform(f_oo);
    bto_random<4, T>().perform(i_oovv);
    bto_random<4, T>().perform(t2);

    letter i, j, k, a, b;

    f2_oo(i|j) = f_oo(i|j) + contract(k|a|b, i_oovv(j|k|a|b), t2(i|k|a|b));

    }

    need_erase = false;
    product_table_container::get_instance().erase(pgtid);

    } catch(exception &e) {
        if(need_erase) {
            product_table_container::get_instance().erase(pgtid);
        }
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_9() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_9()";

    bool need_erase = true;
    const char *pgtid = "point_group_cs";

    try {

    point_group_table::label_t ap = 0, app = 1;
    std::vector<std::string> irnames(2);
    irnames[ap] = "A'"; irnames[app] = "A''";
    point_group_table cs(pgtid, irnames, irnames[ap]);
    cs.add_product(app, app, ap);
    cs.check();
    product_table_container::get_instance().add(cs);

    {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<2> soo(so&so);

    btensor<2, T> f_oo(soo), d1_oo(soo), d2_oo(soo);

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    se_label<2, T> l_oo(soo.get_bis().get_block_index_dims(), pgtid);
    block_labeling<2> &bl_oo = l_oo.get_labeling();
    bl_oo.assign(m11, 0, ap);
    bl_oo.assign(m11, 1, app);
    bl_oo.assign(m11, 2, ap);
    bl_oo.assign(m11, 3, app);
    l_oo.set_rule(ap);

    {
        block_tensor_ctrl<2, T> c_f_oo(f_oo);
        symmetry<2, T> sym_f_oo(f_oo.get_bis());
        scalar_transf<T> tr0;
        sym_f_oo.insert(se_perm<2, T>(permutation<2>().
            permute(0, 1), tr0));
        sym_f_oo.insert(l_oo);
        so_copy<2, T>(sym_f_oo).perform(c_f_oo.req_symmetry());
    }

    bto_copy<2, T>(f_oo).perform(d1_oo);
    bto_set_diag<2, T>(1.0).perform(d1_oo);
    bto_random<2, T>().perform(f_oo);

    letter i, j;

    d2_oo(i|j) = dirsum(diag(i, i|j, f_oo(i|j)), -diag(j, j|i, f_oo(i|j)))
        + d1_oo(i|j);

    }

    need_erase = false;
    product_table_container::get_instance().erase(pgtid);

    } catch(exception &e) {
        if(need_erase) {
            product_table_container::get_instance().erase(pgtid);
        }
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_10() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_10()";

    bool need_erase = true;
    const char *pgtid = "point_group_cs";

    try {

    point_group_table::label_t ap = 0, app = 1;
    std::vector<std::string> irnames(2);
    irnames[ap] = "A'"; irnames[app] = "A''";
    point_group_table cs(pgtid, irnames, irnames[ap]);
    cs.add_product(app, app, ap);
    cs.check();
    product_table_container::get_instance().add(cs);

    {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sx(15); sx.split(4);
    bispace<3> soox((so&so)|sx);
    bispace<4> soooo(so&so&so&so);

    btensor<3, T> b_oox(soox);
    btensor<4, T> i_oooo(soooo);

    mask<3> m110;
    m110[0] = true; m110[1] = true; m110[2] = false;

    se_label<3, T> l_oox(soox.get_bis().get_block_index_dims(), pgtid);
    block_labeling<3> &bl_oox = l_oox.get_labeling();
    bl_oox.assign(m110, 0, ap);
    bl_oox.assign(m110, 1, app);
    bl_oox.assign(m110, 2, ap);
    bl_oox.assign(m110, 3, app);
    l_oox.set_rule(ap);

    {
        block_tensor_ctrl<3, T> c_b_oox(b_oox);
        symmetry<3, T> sym_l_oox(soox.get_bis());
        scalar_transf<T> tr0;
        sym_l_oox.insert(se_perm<3, T>(permutation<3>().
            permute(0, 1), tr0));
        sym_l_oox.insert(l_oox);
        so_copy<3, T>(sym_l_oox).perform(c_b_oox.req_symmetry());
    }

    bto_random<3, T>().perform(b_oox);

    letter p, q, r, s, P;

    i_oooo(p|q|r|s) = asymm(r, s, contract(P, b_oox(p|r|P), b_oox(q|s|P)));

    }

    need_erase = false;
    product_table_container::get_instance().erase(pgtid);

    } catch(exception &e) {
        if(need_erase) {
            product_table_container::get_instance().erase(pgtid);
        }
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_11() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_11()";

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(16); sv.split(8);
    bispace<2> sov(so|sv);
    bispace<4> soooo(so&so&so&so), soovv(so&so|sv&sv), sovov(sov&sov),
        svvvv(sv&sv&sv&sv);

    btensor<2, T> df_ov(sov);
    btensor<4, T> i_oooo(soooo), i_ovov(sovov), i_vvvv(svvvv);
    btensor<4, T> t_oovv(soovv), td2(soovv);

    bto_random<2, T>().perform(df_ov);
    bto_random<4, T>().perform(i_oooo);
    bto_random<4, T>().perform(i_ovov);
    bto_random<4, T>().perform(i_vvvv);
    bto_random<4, T>().perform(t_oovv);

    letter i, j, k, l, a, b, c, d;

    T multiplier1 = 0.5, multiplier2 = 2.0;

    td2(i|j|a|b) = multiplier2 * div(
          asymm(i, j, asymm(a, b,
              contract(k|c, t_oovv(i|k|a|c), i_ovov(k|b|j|c))))
        - multiplier1 * (contract(c|d, t_oovv(i|j|c|d), i_vvvv(a|b|c|d))
                + contract(k|l, i_oooo(i|j|k|l), t_oovv(k|l|a|b))),
          symm(i, j, dirsum(df_ov(i|a), df_ov(j|b))));


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_12() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_12()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2, T> t1(sov), t3(sov), t3_ref(sov);

    bto_random<2, T>().perform(t1);
    bto_random<2, T>().perform(t3_ref);
    bto_copy<2, T>(t3_ref).perform(t3);
    t1.set_immutable();

    bto_copy<2, T>(t1).perform(t3_ref, 1.0);

    letter i, a;

    t3(i|a) += t1(i|a);

    compare_ref_x<2, T>::compare(testname, t3, t3_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_13() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_13()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv), svv(sv&sv);

    btensor<2, T> t1(sov), t2(svv), t3(sov), t3_ref(sov);

    bto_random<2, T>().perform(t1);
    bto_random<2, T>().perform(t2);
    bto_random<2, T>().perform(t3_ref);
    bto_copy<2, T>(t3_ref).perform(t3);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 1, 1> contr;
    contr.contract(1, 1);
    bto_contract2<1, 1, 1, T>(contr, t1, t2).perform(t3_ref, 1.0);

    letter i, a, b;

    t3(i|a) += contract(b, t1(i|b), t2(a|b));

    compare_ref_x<2, T>::compare(testname, t3, t3_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void expr_test_x<T>::test_14() throw(libtest::test_exception) {

    static const char *testname = "expr_test_x<T>::test_14()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2, T> t1(sov), t1_ref(sov);

    bto_random<2, T>().perform(t1);

    bto_copy<2, T>(t1).perform(t1_ref);
    bto_scale<2, T>(t1_ref, 0.4).perform();

    letter i, a;

    t1(i|a) *= 0.4;

    compare_ref_x<2, T>::compare(testname, t1, t1_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
