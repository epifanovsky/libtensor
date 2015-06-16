#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/libtensor.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include "../compare_ref.h"
#include "ctf_expr_test.h"

namespace libtensor {


void ctf_expr_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);
    ctf::init();

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();
        test_6();

    } catch(...) {
        ctf::exit();
        allocator<double>::shutdown();
        throw;
    }

    ctf::exit();
    allocator<double>::shutdown();
}


void ctf_expr_test::test_1() {

    static const char testname[] = "ctf_expr_test::test_1()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2, double> t1(sov), t2(sov), t3(sov), t3_ref(sov);
    ctf_btensor<2, double> dt1(sov), dt2(sov), dt3(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);

    letter i, a;

    t3_ref(i|a) = t1(i|a) - t2(i|a);
    dt3(i|a) = dt1(i|a) - dt2(i|a);

    ctf_btod_collect<2>(dt3).perform(t3);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_expr_test::test_2() {

    static const char testname[] = "ctf_expr_test::test_2()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    btensor<2, double> t1(sov);
    btensor<4, double> t2(soovv);
    btensor<2, double> f_ov(sov);
    btensor<4, double> i_ooov(sooov), i_oovv(soovv), i_ovov(sovov),
        i_ovvv(sovvv);
    btensor<4, double> i3_ovvv(sovvv), i3_ovvv_ref(sovvv);
    ctf_btensor<2, double> dt1(sov);
    ctf_btensor<4, double> dt2(soovv);
    ctf_btensor<2, double> df_ov(sov);
    ctf_btensor<4, double> di_ooov(sooov), di_oovv(soovv), di_ovov(sovov),
        di_ovvv(sovvv);
    ctf_btensor<4, double> di3_ovvv(sovvv);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<2>().perform(f_ov);
    btod_random<4>().perform(i_ooov);
    btod_random<4>().perform(i_oovv);
    btod_random<4>().perform(i_ovov);
    btod_random<4>().perform(i_ovvv);

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<4>(t2).perform(dt2);
    ctf_btod_distribute<2>(f_ov).perform(df_ov);
    ctf_btod_distribute<4>(i_ooov).perform(di_ooov);
    ctf_btod_distribute<4>(i_oovv).perform(di_oovv);
    ctf_btod_distribute<4>(i_ovov).perform(di_ovov);
    ctf_btod_distribute<4>(i_ovvv).perform(di_ovvv);

    letter i, j, k, a, b, c, d;

    i3_ovvv_ref(i|a|b|c) =
          i_ovvv(i|a|b|c)
        + asymm(b, c, contract(j,
            t1(j|c),
            i_ovov(j|b|i|a)
            - contract(k|d, t2(i|k|b|d), i_oovv(j|k|a|d))))
        - asymm(b, c, contract(k|d, i_ovvv(k|c|a|d), t2(i|k|b|d)));

    di3_ovvv(i|a|b|c) =
          di_ovvv(i|a|b|c)
        + asymm(b, c, contract(j,
            dt1(j|c),
            di_ovov(j|b|i|a)
            - contract(k|d, dt2(i|k|b|d), di_oovv(j|k|a|d))))
        - asymm(b, c, contract(k|d, di_ovvv(k|c|a|d), dt2(i|k|b|d)));

    ctf_btod_collect<4>(di3_ovvv).perform(i3_ovvv);

    compare_ref<4>::compare(testname, i3_ovvv, i3_ovvv_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_expr_test::test_3() {

    static const char testname[] = "ctf_expr_test::test_3()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2, double> t1(sov), t3(sov), t3_ref(sov);
    ctf_btensor<2, double> dt1(sov), dt3(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t3_ref);
    btod_copy<2>(t3_ref).perform(t3);
    t1.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t3).perform(dt3);

    letter i, a;

    t3_ref(i|a) += t1(i|a);
    dt3(i|a) += dt1(i|a);

    ctf_btod_collect<2>(dt3).perform(t3);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_expr_test::test_4() {

    static const char testname[] = "ctf_expr_test::test_4()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv), svv(sv&sv);

    btensor<2, double> t1(sov), t2(svv), t3(sov), t3_ref(sov);
    ctf_btensor<2, double> dt1(sov), dt2(svv), dt3(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    btod_random<2>().perform(t3_ref);
    btod_copy<2>(t3_ref).perform(t3);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);
    ctf_btod_distribute<2>(t3).perform(dt3);

    letter i, a, b;

    t3_ref(i|a) += contract(b, t1(i|b), t2(a|b));
    dt3(i|a) += contract(b, dt1(i|b), dt2(a|b));

    ctf_btod_collect<2>(dt3).perform(t3);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_expr_test::test_5() {

    static const char testname[] = "ctf_expr_test::test_5()";

    try {

    bispace<1> so(13); so.split(7);
    bispace<1> sv(7); sv.split(3);

    bispace<2> sov(so|sv);
    bispace<4> sooov((so&so&so)|sv), soovv((so&so)|(sv&sv)),
		sovvv(so|(sv&sv&sv)), svvvv(sv&sv&sv&sv);

    bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
    bispace<4> sovov(so1|sv1|so2|sv2, (so1&so2)|(sv1&sv2));

    symmetry<2, double> sym_ov(sov.get_bis());
    symmetry<4, double> sym_oovv(soovv.get_bis()), sym_ovvv(sovvv.get_bis()),
        sym_vvvv(svvvv.get_bis());
    {
        permutation<4> p1032;
        p1032.permute(0, 1).permute(2, 3);
        se_perm<4, double> sep1032(p1032, scalar_transf<double>());
        sym_oovv.insert(sep1032);
        sym_vvvv.insert(sep1032);
    }

    btensor<2, double> t1(sov);
    btensor<4, double> t2(soovv);
    btensor<4, double> i_oovv(soovv), i_ovvv(sovvv), i_vvvv(svvvv);
    btensor<4, double> i5_vvvv(svvvv), i5_vvvv_ref(svvvv);
    ctf_btensor<2, double> dt1(sov);
    ctf_btensor<4, double> dt2(soovv);
    ctf_btensor<4, double> di_oovv(soovv), di_ovvv(sovvv), di_vvvv(svvvv);
    ctf_btensor<4, double> di5_vvvv(svvvv);

    {
        block_tensor_ctrl<4, double> ctrl_t2(t2), ctrl_i_oovv(i_oovv),
            ctrl_i_vvvv(i_vvvv);
        so_copy<4, double>(sym_oovv).perform(ctrl_t2.req_symmetry());
        so_copy<4, double>(sym_oovv).perform(ctrl_i_oovv.req_symmetry());
        so_copy<4, double>(sym_vvvv).perform(ctrl_i_vvvv.req_symmetry());
    }

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<4>().perform(i_oovv);
    btod_random<4>().perform(i_ovvv);
    btod_random<4>().perform(i_vvvv);

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<4>(t2).perform(dt2);
    ctf_btod_distribute<4>(i_oovv).perform(di_oovv);
    ctf_btod_distribute<4>(i_ovvv).perform(di_ovvv);
    ctf_btod_distribute<4>(i_vvvv).perform(di_vvvv);

    letter i, j, k, a, b, c, d;

    i5_vvvv_ref(a|b|c|d) =
          i_vvvv(a|b|c|d)
        + 0.5 * contract(i|j, t2(i|j|a|b), i_oovv(i|j|c|d));
        - asymm(a, b, contract(i, i_ovvv(i|b|c|d), t1(i|a)));

    di5_vvvv(a|b|c|d) =
          di_vvvv(a|b|c|d)
        + 0.5 * contract(i|j, dt2(i|j|a|b), di_oovv(i|j|c|d));
        - asymm(a, b, contract(i, di_ovvv(i|b|c|d), dt1(i|a)));

    ctf_btod_collect<4>(di5_vvvv).perform(i5_vvvv);

    compare_ref<4>::compare(testname, i5_vvvv, i5_vvvv_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_expr_test::test_6() {

    static const char testname[] = "ctf_expr_test::test_6()";

    try {

    bispace<1> so(10); so.split(2).split(5).split(7);
    bispace<1> sv(4); sv.split(1).split(2).split(3);

    bispace<2> sov(so|sv);
    bispace<4> soooo(so&so&so&so), sooov((so&so&so)|sv), soovv((so&so)|(sv&sv));

    symmetry<2, double> sym_ov(sov.get_bis());
    symmetry<4, double> sym_oooo(soooo.get_bis()), sym_ooov(sooov.get_bis()),
        sym_oovv(soovv.get_bis());
    {
        permutation<4> p1032;
        p1032.permute(0, 1).permute(2, 3);
        se_perm<4, double> sep1032(p1032, scalar_transf<double>());
        sym_oooo.insert(sep1032);
        sym_oovv.insert(sep1032);
    }

    btensor<2, double> t1(sov);
    btensor<4, double> t2(soovv);
    btensor<4, double> i_oooo(soooo), i_ooov(sooov), i_oovv(soovv);
    btensor<4, double> i4_oooo(soooo), i4_oooo_ref(soooo);
    ctf_btensor<2, double> dt1(sov);
    ctf_btensor<4, double> dt2(soovv);
    ctf_btensor<4, double> di_oooo(soooo), di_ooov(sooov), di_oovv(soovv);
    ctf_btensor<4, double> di4_oooo(soooo);

    {
        block_tensor_ctrl<4, double> ctrl_i_oooo(i_oooo), ctrl_i_oovv(i_oovv),
            ctrl_t2(t2);
        so_copy<4, double>(sym_oooo).perform(ctrl_i_oooo.req_symmetry());
        so_copy<4, double>(sym_oovv).perform(ctrl_i_oovv.req_symmetry());
        so_copy<4, double>(sym_oovv).perform(ctrl_t2.req_symmetry());
    }

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<4>().perform(i_oooo);
    btod_random<4>().perform(i_ooov);
    btod_random<4>().perform(i_oovv);

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<4>(t2).perform(dt2);
    ctf_btod_distribute<4>(i_oooo).perform(di_oooo);
    ctf_btod_distribute<4>(i_ooov).perform(di_ooov);
    ctf_btod_distribute<4>(i_oovv).perform(di_oovv);

    letter i, j, k, l, a, b, c, d;

    i4_oooo_ref(i|j|k|l) =
          i_oooo(i|j|k|l)
        + 0.5 * contract(a|b, i_oovv(k|l|a|b), t2(i|j|a|b))
        + asymm(i, j, contract(a, i_ooov(k|l|i|a), t1(j|a)));
    di4_oooo(i|j|k|l) =
          di_oooo(i|j|k|l)
        + 0.5 * contract(a|b, di_oovv(k|l|a|b), dt2(i|j|a|b))
        + asymm(i, j, contract(a, di_ooov(k|l|i|a), dt1(j|a)));

    ctf_btod_collect<4>(di4_oooo).perform(i4_oooo);

    compare_ref<4>::compare(testname, i4_oooo, i4_oooo_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
