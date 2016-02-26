#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/libtensor.h>
#include "../compare_ref.h"
#include "expr_test.h"

namespace libtensor {


void expr_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

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
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void expr_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_1()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2> t1(sov), t2(sov), t3(sov), t3_ref(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_add<2> op(t1);
    op.add_op(t2, -1.0);
    op.perform(t3_ref);

    letter i, a;

    t3(i|a) = t1(i|a) - t2(i|a);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_2()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<2> t1(sov);
    btensor<4> t2(soovv);
    btensor<2> f_ov(sov);
    btensor<4> i_ooov(sooov), i_oovv(soovv), i_ovov(sovov), i_ovvv(sovvv);
    btensor<4> i3_ovvv(sovvv), i5_vvvv(svvvv);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<2>().perform(f_ov);
    btod_random<4>().perform(i_ooov);
    btod_random<4>().perform(i_oovv);
    btod_random<4>().perform(i_ovov);
    btod_random<4>().perform(i_ovvv);
    btod_random<4>().perform(i5_vvvv);

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


void expr_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_3()";

    try {

    bispace<1> so(10); so.split(5);
    bispace<1> sv(4); sv.split(2);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<2> t1(sov);
    btensor<4> t2(soovv);
    btensor<4> i1_ovov(sovov);

    {
        block_tensor_ctrl<4, double> c_t2(t2);
        symmetry<4, double> sym_t2(t2.get_bis());
        scalar_transf<double> tr1(-1.);
        sym_t2.insert(se_perm<4, double>(permutation<4>().
            permute(0, 1), tr1));
        sym_t2.insert(se_perm<4, double>(permutation<4>().
            permute(2, 3), tr1));
        so_copy<4, double>(sym_t2).perform(c_t2.req_symmetry());
    }

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<4>().perform(i1_ovov);

    letter i, k, b, c;

    i1_ovov(i|b|k|c) = t2(i|k|b|c) - t1(i|c) * t1(k|b);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_4()";

    try {

    bispace<1> so(10); so.split(5);
    bispace<1> sv(4); sv.split(2);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<4> i_oovv(soovv), i_ovov(sovov), t2(soovv);
    btensor<4> i2_oovv(soovv);

    {
        block_tensor_ctrl<4, double> c_i_oovv(i_oovv), c_t2(t2);
        scalar_transf<double> tr1(-1.);
        symmetry<4, double> sym_t2(t2.get_bis());
        sym_t2.insert(se_perm<4, double>(permutation<4>().
            permute(0, 1), tr1));
        sym_t2.insert(se_perm<4, double>(permutation<4>().
            permute(2, 3), tr1));
        so_copy<4, double>(sym_t2).perform(c_t2.req_symmetry());
        so_copy<4, double>(sym_t2).perform(c_i_oovv.req_symmetry());
    }
    {
        scalar_transf<double> tr0;
        block_tensor_ctrl<4, double> c_i_ovov(i_ovov);
        symmetry<4, double> sym_i_ovov(i_ovov.get_bis());
        sym_i_ovov.insert(se_perm<4, double>(permutation<4>().
            permute(0, 2).permute(1, 3), tr0));
        so_copy<4, double>(sym_i_ovov).perform(c_i_ovov.req_symmetry());
    }

    btod_random<4>().perform(i_oovv);
    btod_random<4>().perform(i_ovov);
    btod_random<4>().perform(t2);

    letter j, k, l, a, b, c;

    i2_oovv(j|k|a|b) =
        i_ovov(k|a|j|b) - contract(l|c, i_oovv(k|l|b|c), t2(j|l|a|c));

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_5()";

    try {

    bispace<1> so(10); so.split(5);
    bispace<1> sv(4); sv.split(2);

    bispace<2> soo(so&so);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    btensor<4> t1_oovv(soovv), t2_oovv(soovv);
    btensor<2> t3_oo(soo), t3_oo_ref(soo);

    {
        scalar_transf<double> tr1(-1.);
        block_tensor_ctrl<4, double> c_t1_oovv(t1_oovv);
        c_t1_oovv.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        c_t1_oovv.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }

    btod_random<4>().perform(t1_oovv);
    btod_random<4>().perform(t2_oovv);

    letter i, j, k, a, b;

    t3_oo(i|j) = contract(k|a|b, t1_oovv(j|k|a|b), t2_oovv(i|k|a|b));

    contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
    contr.contract(1, 1);
    contr.contract(2, 2);
    contr.contract(3, 3);
    btod_contract2<1, 1, 3>(contr, t1_oovv, t2_oovv).perform(t3_oo_ref);
    compare_ref<2>::compare(testname, t3_oo, t3_oo_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_6()";

    typedef allocator<double> allocator_t;

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(4); sv.split(1).split(2).split(3);

    bispace<2> sov(so|sv);
    bispace<4> soooo(so&so&so&so), sooov((so&so&so)|sv), soovv((so&so)|(sv&sv));

    btensor<4> i_oooo(soooo), i_ooov(sooov), i_oovv(soovv);
    btensor<2> t1(sov);
    btensor<4> t2(soovv);
    btensor<4> i4_oooo(soooo), i4_oooo_ref(soooo);

    {
        scalar_transf<double> tr1(-1.);
        block_tensor_ctrl<4, double> c_i_oooo(i_oooo);
        c_i_oooo.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        c_i_oooo.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));

        block_tensor_ctrl<4, double> c_i_ooov(i_ooov);
        c_i_ooov.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));

        block_tensor_ctrl<4, double> c_i_oovv(i_oovv);
        c_i_oovv.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        c_i_oovv.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));

        block_tensor_ctrl<4, double> c_t2(t2);
        c_t2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        c_t2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }

    btod_random<4>().perform(i_oooo);
    btod_random<4>().perform(i_ooov);
    btod_random<4>().perform(i_oovv);
    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);

    letter i, j, k, l, a, b;

    i4_oooo(i|j|k|l) =
          i_oooo(i|j|k|l)
        + 0.5 * contract(a|b, i_oovv(k|l|a|b), t2(i|j|a|b))
        + asymm(i, j, contract(a, i_ooov(k|l|i|a), t1(j|a)))
        + contract(a|b, i_oovv(k|l|a|b), t1(i|a)*t1(j|b));

    btod_copy<4> op1(i_oooo);

    contraction2<2, 2, 2> contr2;
    contr2.contract(2, 2);
    contr2.contract(3, 3);
    btod_contract2<2, 2, 2> op2(contr2, t2, i_oovv);

    contraction2<3, 1, 1> contr3(permutation<4>().
        permute(0, 2).permute(1, 3));
    contr3.contract(3, 1);
    btod_contract2<3, 1, 1> op3a(contr3, i_ooov, t1);
    btod_symmetrize2<4> op3(op3a, 0, 1, false);

    contraction2<2, 2, 0> contr4a(permutation<4>().permute(1, 2));
    btensor<4> tmp4a(soovv);
    btod_contract2<2, 2, 0>(contr4a, t1, t1).perform(tmp4a);
    contraction2<2, 2, 2> contr4;
    contr4.contract(2, 2);
    contr4.contract(3, 3);
    btod_contract2<2, 2, 2> op4(contr4, tmp4a, i_oovv);

    op1.perform(i4_oooo_ref);
    op2.perform(i4_oooo_ref, 0.5);
    op3.perform(i4_oooo_ref, 1.0);
    op4.perform(i4_oooo_ref, 1.0);

    compare_ref<4>::compare(testname, i4_oooo, i4_oooo_ref, 5e-15);

    dense_tensor<4, double, allocator_t> ti_oooo(i_oooo.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> ti_ooov(i_ooov.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> ti_oovv(i_oovv.get_bis().get_dims());
    dense_tensor<2, double, allocator_t> tt1(t1.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> tt2(t2.get_bis().get_dims());
    tod_btconv<4>(i_oooo).perform(ti_oooo);
    tod_btconv<4>(i_ooov).perform(ti_ooov);
    tod_btconv<4>(i_oovv).perform(ti_oovv);
    tod_btconv<2>(t1).perform(tt1);
    tod_btconv<4>(t2).perform(tt2);

    tod_copy<4> top1(ti_oooo);

    tod_contract2<2, 2, 2> top2(contr2, tt2, ti_oovv, 0.5);

    dense_tensor<4, double, allocator_t> ttmp3a(i_oooo.get_bis().get_dims());
    tod_contract2<3, 1, 1>(contr3, ti_ooov, tt1).perform(true, ttmp3a);
    tod_add<4> top3(ttmp3a);
    top3.add_op(ttmp3a, permutation<4>().permute(0, 1), -1.0);

    dense_tensor<4, double, allocator_t> ttmp4a(i_oovv.get_bis().get_dims());
    tod_contract2<2, 2, 0>(contr4a, tt1, tt1).perform(true, ttmp4a);
    tod_contract2<2, 2, 2> top4(contr4, ttmp4a, ti_oovv);

    dense_tensor<4, double, allocator_t> ti4_oooo(i4_oooo.get_bis().get_dims()),
        ti4_oooo_ref(i4_oooo.get_bis().get_dims());
    top1.perform(true, ti4_oooo_ref);
    top2.perform(false, ti4_oooo_ref);
    top3.perform(false, ti4_oooo_ref);
    top4.perform(false, ti4_oooo_ref);
    tod_btconv<4>(i4_oooo).perform(ti4_oooo);

    compare_ref<4>::compare(testname, ti4_oooo, ti4_oooo_ref, 5e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_7() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_7()";

    typedef allocator<double> allocator_t;

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(4); sv.split(1).split(2).split(3);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)), sovvv(so|(sv&sv&sv));

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<4> i_ooov(sooov), i_oovv(soovv), i_ovov(sovov), i_ovvv(sovvv);
    btensor<2> t1(sov);
    btensor<4> t2(soovv);
    btensor<4> i1_ovov(sovov), i1_ovov_ref(sovov);

    btod_random<4>().perform(i_ooov);
    btod_random<4>().perform(i_oovv);
    btod_random<4>().perform(i_ovov);
    btod_random<4>().perform(i_ovvv);
    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);

    letter i, j, k, a, b, c;

    i1_ovov(i|a|j|b) =
          i_ovov(i|a|j|b)
        - contract(c, i_ovvv(i|a|b|c), t1(j|c))
        - contract(k, i_ooov(i|k|j|b), t1(k|a))
        + 0.5 * contract(k|c,
                          t2(j|k|c|a) + 2.0 * t1(j|c)*t1(k|a),
                               i_oovv(i|k|b|c));

    btod_copy<4> op1(i_ovov);

    contraction2<3, 1, 1> contr2(permutation<4>().permute(2, 3));
    contr2.contract(3, 1);
    btod_contract2<3, 1, 1> op2(contr2, i_ovvv, t1);

    contraction2<3, 1, 1> contr3(permutation<4>().
        permute(1, 3).permute(2, 3));
    contr3.contract(1, 0);
    btod_contract2<3, 1, 1> op3(contr3, i_ooov, t1);

    contraction2<2, 2, 0> contr4a(permutation<4>().permute(1, 2));
    btensor<4> tmp4a(soovv);
    btod_copy<4>(t2).perform(tmp4a);
    btod_contract2<2, 2, 0>(contr4a, t1, t1).perform(tmp4a, 2.0);

    contraction2<2, 2, 2> contr4(permutation<4>().permute(0, 2));
    contr4.contract(1, 1);
    contr4.contract(2, 3);
    btod_contract2<2, 2, 2> op4(contr4, tmp4a, i_oovv);

    op1.perform(i1_ovov_ref);
    op2.perform(i1_ovov_ref, -1.0);
    op3.perform(i1_ovov_ref, -1.0);
    op4.perform(i1_ovov_ref, 0.5);

    compare_ref<4>::compare(testname, i1_ovov, i1_ovov_ref, 5e-15);

    dense_tensor<4, double, allocator_t> ti_ooov(i_ooov.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> ti_oovv(i_oovv.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> ti_ovov(i_ovov.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> ti_ovvv(i_ovvv.get_bis().get_dims());
    dense_tensor<2, double, allocator_t> tt1(t1.get_bis().get_dims());
    dense_tensor<4, double, allocator_t> tt2(t2.get_bis().get_dims());
    tod_btconv<4>(i_ooov).perform(ti_ooov);
    tod_btconv<4>(i_oovv).perform(ti_oovv);
    tod_btconv<4>(i_ovov).perform(ti_ovov);
    tod_btconv<4>(i_ovvv).perform(ti_ovvv);
    tod_btconv<2>(t1).perform(tt1);
    tod_btconv<4>(t2).perform(tt2);

    tod_copy<4> top1(ti_ovov);

    tod_contract2<3, 1, 1> top2(contr2, ti_ovvv, tt1, -1.0);

    tod_contract2<3, 1, 1> top3(contr3, ti_ooov, tt1, -1.0);

    dense_tensor<4, double, allocator_t> ttmp4a(i_oovv.get_bis().get_dims());
    tod_copy<4>(tt2).perform(true, ttmp4a);
    tod_contract2<2, 2, 0>(contr4a, tt1, tt1, 2.0).perform(false, ttmp4a);
    tod_contract2<2, 2, 2> top4(contr4, ttmp4a, ti_oovv, 0.5);

    dense_tensor<4, double, allocator_t> ti1_ovov(i1_ovov.get_bis().get_dims()),
        ti1_ovov_ref(i1_ovov.get_bis().get_dims());
    top1.perform(true, ti1_ovov_ref);
    top2.perform(false, ti1_ovov_ref);
    top3.perform(false, ti1_ovov_ref);
    top4.perform(false, ti1_ovov_ref);
    tod_btconv<4>(i1_ovov).perform(ti1_ovov);

    compare_ref<4>::compare(testname, ti1_ovov, ti1_ovov_ref, 5e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_8() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_8()";

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

    btensor<4> i_oovv(soovv), t2(soovv);
    btensor<2> f_oo(soo), f2_oo(soo);

    mask<2> m11;
    m11[0] = true; m11[1] = true;
    mask<4> m0011, m1100;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;

    se_label<2, double> l_oo(soo.get_bis().get_block_index_dims(), pgtid);
    block_labeling<2> &bl_oo = l_oo.get_labeling();
    bl_oo.assign(m11, 0, ap);
    bl_oo.assign(m11, 1, app);
    bl_oo.assign(m11, 2, ap);
    bl_oo.assign(m11, 3, app);
    l_oo.set_rule(ap);
    se_label<4, double> l_oovv(soovv.get_bis().get_block_index_dims(),
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
        block_tensor_ctrl<2, double> c_f_oo(f_oo);
        symmetry<2, double> sym_f_oo(f_oo.get_bis());
        scalar_transf<double> tr0, tr1(-1.);
        sym_f_oo.insert(se_perm<2, double>(permutation<2>().
            permute(0, 1), tr0));
        sym_f_oo.insert(l_oo);
        so_copy<2, double>(sym_f_oo).perform(c_f_oo.req_symmetry());

        block_tensor_ctrl<4, double> c_i_oovv(i_oovv), c_t2(t2);
        symmetry<4, double> sym_t2(t2.get_bis());
        sym_t2.insert(se_perm<4, double>(permutation<4>().
            permute(0, 1), tr1));
        sym_t2.insert(se_perm<4, double>(permutation<4>().
            permute(2, 3), tr1));
        sym_t2.insert(l_oovv);
        so_copy<4, double>(sym_t2).perform(c_t2.req_symmetry());
        so_copy<4, double>(sym_t2).perform(c_i_oovv.req_symmetry());
    }

    btod_random<2>().perform(f_oo);
    btod_random<4>().perform(i_oovv);
    btod_random<4>().perform(t2);

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


void expr_test::test_9() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_9()";

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

    btensor<2> f_oo(soo), d1_oo(soo), d2_oo(soo);

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    se_label<2, double> l_oo(soo.get_bis().get_block_index_dims(), pgtid);
    block_labeling<2> &bl_oo = l_oo.get_labeling();
    bl_oo.assign(m11, 0, ap);
    bl_oo.assign(m11, 1, app);
    bl_oo.assign(m11, 2, ap);
    bl_oo.assign(m11, 3, app);
    l_oo.set_rule(ap);

    {
        block_tensor_ctrl<2, double> c_f_oo(f_oo);
        symmetry<2, double> sym_f_oo(f_oo.get_bis());
        scalar_transf<double> tr0;
        sym_f_oo.insert(se_perm<2, double>(permutation<2>().
            permute(0, 1), tr0));
        sym_f_oo.insert(l_oo);
        so_copy<2, double>(sym_f_oo).perform(c_f_oo.req_symmetry());
    }

    btod_copy<2>(f_oo).perform(d1_oo);
    btod_set_diag<2>(1.0).perform(d1_oo);
    btod_random<2>().perform(f_oo);

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


void expr_test::test_10() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_10()";

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

    btensor<3, double> b_oox(soox);
    btensor<4, double> i_oooo(soooo);

    mask<3> m110;
    m110[0] = true; m110[1] = true; m110[2] = false;

    se_label<3, double> l_oox(soox.get_bis().get_block_index_dims(), pgtid);
    block_labeling<3> &bl_oox = l_oox.get_labeling();
    bl_oox.assign(m110, 0, ap);
    bl_oox.assign(m110, 1, app);
    bl_oox.assign(m110, 2, ap);
    bl_oox.assign(m110, 3, app);
    l_oox.set_rule(ap);

    {
        block_tensor_ctrl<3, double> c_b_oox(b_oox);
        symmetry<3, double> sym_l_oox(soox.get_bis());
        scalar_transf<double> tr0;
        sym_l_oox.insert(se_perm<3, double>(permutation<3>().
            permute(0, 1), tr0));
        sym_l_oox.insert(l_oox);
        so_copy<3, double>(sym_l_oox).perform(c_b_oox.req_symmetry());
    }

    btod_random<3>().perform(b_oox);

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


void expr_test::test_11() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_11()";

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(16); sv.split(8);
    bispace<2> sov(so|sv);
    bispace<4> soooo(so&so&so&so), soovv(so&so|sv&sv), sovov(sov&sov),
        svvvv(sv&sv&sv&sv);

    btensor<2, double> df_ov(sov);
    btensor<4, double> i_oooo(soooo), i_ovov(sovov), i_vvvv(svvvv);
    btensor<4, double> t_oovv(soovv), td2(soovv);

    btod_random<2>().perform(df_ov);
    btod_random<4>().perform(i_oooo);
    btod_random<4>().perform(i_ovov);
    btod_random<4>().perform(i_vvvv);
    btod_random<4>().perform(t_oovv);

    letter i, j, k, l, a, b, c, d;

    td2(i|j|a|b) = 2.0 * div(
          asymm(i, j, asymm(a, b,
              contract(k|c, t_oovv(i|k|a|c), i_ovov(k|b|j|c))))
        - 0.5 * (contract(c|d, t_oovv(i|j|c|d), i_vvvv(a|b|c|d))
                + contract(k|l, i_oooo(i|j|k|l), t_oovv(k|l|a|b))),
          symm(i, j, dirsum(df_ov(i|a), df_ov(j|b))));


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_12() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_12()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2> t1(sov), t3(sov), t3_ref(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t3_ref);
    btod_copy<2>(t3_ref).perform(t3);
    t1.set_immutable();

    btod_copy<2>(t1).perform(t3_ref, 1.0);

    letter i, a;

    t3(i|a) += t1(i|a);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_13() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_13()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv), svv(sv&sv);

    btensor<2> t1(sov), t2(svv), t3(sov), t3_ref(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    btod_random<2>().perform(t3_ref);
    btod_copy<2>(t3_ref).perform(t3);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 1, 1> contr;
    contr.contract(1, 1);
    btod_contract2<1, 1, 1>(contr, t1, t2).perform(t3_ref, 1.0);

    letter i, a, b;

    t3(i|a) += contract(b, t1(i|b), t2(a|b));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_test::test_14() throw(libtest::test_exception) {

    static const char *testname = "expr_test::test_14()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2> t1(sov), t1_ref(sov);

    btod_random<2>().perform(t1);

    btod_copy<2>(t1).perform(t1_ref);
    btod_scale<2>(t1_ref, 0.4).perform();

    letter i, a;

    t1(i|a) *= 0.4;

    compare_ref<2>::compare(testname, t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
