#include <libtensor/core/allocator.h>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/iface/iface.h>
#include <libtensor/iface/expr/direct_eval.h>
#include "../compare_ref.h"
#include "direct_eval_test.h"

namespace libtensor {


void direct_eval_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

        test_copy_1();
        test_copy_2();
        test_copy_3();
        test_copy_4();
        test_copy_5();
        test_copy_6();

        test_add_1();

        test_contr_1();
        test_contr_2();

        test_mixed_1();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


template<size_t N, typename T, typename Core>
void direct_eval_test::invoke_eval(
    const char *testname,
    const labeled_btensor_expr::expr<N, T, Core> &expr,
    const letter_expr<N> &label, block_tensor_i<N, T> &ref, double thresh)
    throw(libtest::test_exception) {

    try {

    labeled_btensor_expr::direct_eval<N, T, Core> ev(expr, label);
    btensor<N, T> bt(ev.get_btensor().get_bis());
    btod_copy<N>(ev.get_btensor()).perform(bt);
    compare_ref<N>::compare(testname, bt, ref, thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_copy_1() throw(libtest::test_exception) {

    //
    //  Simple copy, no symmetry
    //  q(i|j|a|b) = p(i|j|a|b)
    //

    static const char *testname = "direct_eval_test::test_copy_1()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si|sj|sa|sb);

    btensor<4> tp(sijab);
    btod_random<4>().perform(tp);

    block_tensor<4, double, allocator_t> tp_ref(sijab.get_bis());
    btod_copy<4>(tp).perform(tp_ref);

    letter i, j, a, b;
    invoke_eval(testname, 1.0*tp(i|j|a|b), i|j|a|b, tp_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_copy_2() throw(libtest::test_exception) {

    //
    //  Permuted copy, no symmetry
    //  q(i|a|j|b) = p(i|j|a|b)
    //

    static const char *testname = "direct_eval_test::test_copy_2()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si|sj|sa|sb), siajb(si|sa|sj|sb);

    btensor<4> tp(sijab);
    btod_random<4>().perform(tp);

    block_tensor<4, double, allocator_t> tp_ref(siajb.get_bis());
    permutation<4> perm;
    perm.permute(1, 2); // ijab -> iajb
    btod_copy<4>(tp, perm).perform(tp_ref);

    letter i, j, a, b;
    invoke_eval(testname, 1.0*tp(i|j|a|b), i|a|j|b, tp_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_copy_3() throw(libtest::test_exception) {

    //
    //  Scaled copy, no symmetry
    //  q(i|j|a|b) = p(i|j|a|b)*1.5
    //

    static const char *testname = "direct_eval_test::test_copy_3()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si|sj|sa|sb);

    btensor<4> tp(sijab);
    btod_random<4>().perform(tp);

    block_tensor<4, double, allocator_t> tp_ref(sijab.get_bis());
    btod_copy<4>(tp, 1.5).perform(tp_ref);

    letter i, j, a, b;
    invoke_eval(testname, tp(i|j|a|b)*1.5, i|j|a|b, tp_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_copy_4() throw(libtest::test_exception) {

    //
    //  Scaled permuted copy, no symmetry
    //  q(i|a|j|b) = -1.5*p(i|j|a|b)
    //

    static const char *testname = "direct_eval_test::test_copy_4()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si|sj|sa|sb), siajb(si|sa|sj|sb);

    btensor<4> tp(sijab);
    btod_random<4>().perform(tp);

    block_tensor<4, double, allocator_t> tp_ref(siajb.get_bis());
    permutation<4> perm;
    perm.permute(1, 2); // ijab -> iajb
    btod_copy<4>(tp, perm, -1.5).perform(tp_ref);

    letter i, j, a, b;
    invoke_eval(testname, -1.5*tp(i|j|a|b), i|a|j|b, tp_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_copy_5() throw(libtest::test_exception) {

    //
    //  Simple copy, permutational symmetry
    //  q(i|j|a|b) = p(i|j|a|b)
    //

    static const char *testname = "direct_eval_test::test_copy_5()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si&sj|sa&sb);

    btensor<4> tp(sijab);
    scalar_transf<double> tr0;
    se_perm<4, double> cycle1(permutation<4>().permute(0, 1), tr0);
    se_perm<4, double> cycle2(permutation<4>().permute(2, 3), tr0);
    block_tensor_ctrl<4, double> ctrl(tp);
    ctrl.req_symmetry().insert(cycle1);
    ctrl.req_symmetry().insert(cycle2);
    btod_random<4>().perform(tp);

    block_tensor<4, double, allocator_t> tp_ref(sijab.get_bis());
    btod_copy<4>(tp).perform(tp_ref);

    letter i, j, a, b;
    invoke_eval(testname, 1.0*tp(i|j|a|b), i|j|a|b, tp_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_copy_6() throw(libtest::test_exception) {

    //
    //  Permuted copy, permutational symmetry
    //  q(i|a|j|b) = p(i|j|a|b)
    //

    static const char *testname = "direct_eval_test::test_copy_6()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si&sj|sa&sb), siajb(si|sa|sj|sb, si&sj|sa&sb);

    btensor<4> tp(sijab);
    scalar_transf<double> tr0;
    se_perm<4, double> cycle1(permutation<4>().permute(0, 1), tr0);
    se_perm<4, double> cycle2(permutation<4>().permute(2, 3), tr0);
    block_tensor_ctrl<4, double> ctrl(tp);
    ctrl.req_symmetry().insert(cycle1);
    ctrl.req_symmetry().insert(cycle2);
    btod_random<4>().perform(tp);

    block_tensor<4, double, allocator_t> tp_ref(siajb.get_bis());
    permutation<4> perm;
    perm.permute(1, 2); // ijab -> iajb
    btod_copy<4>(tp, perm).perform(tp_ref);

    letter i, j, a, b;
    invoke_eval(testname, 1.0*tp(i|j|a|b), i|a|j|b, tp_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_add_1() throw(libtest::test_exception) {

    //
    //  Addition of two tensors, no symmetry
    //  r(i|j|a|b) = p(i|j|a|b) + q(i|j|a|b)
    //

    static const char *testname = "direct_eval_test::test_add_1()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si&sj|sa&sb);
    btensor<4> tp(sijab), tq(sijab);
    btod_random<4>().perform(tp);
    btod_random<4>().perform(tq);

    block_tensor<4, double, allocator_t> tr_ref(sijab.get_bis());
    btod_add<4> add(tp);
    add.add_op(tq);
    add.perform(tr_ref);

    letter i, j, a, b;
    invoke_eval(testname, tp(i|j|a|b) + tq(i|j|a|b), i|j|a|b, tr_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_contr_1() throw(libtest::test_exception) {

    //
    //  Contraction of two tensors, no symmetry
    //  r(i|j|k|l) = p(i|j|a|b) * q(k|l|a|b)
    //

    static const char *testname = "direct_eval_test::test_contr_1()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si&sj|sa&sb), sijkl(si&sj&si&sj);

    btensor<4> tp(sijab), tq(sijab);
    btod_random<4>().perform(tp);
    btod_random<4>().perform(tq);

    block_tensor<4, double, allocator_t> tr_ref(sijkl.get_bis());
    contraction2<2, 2, 2> contr;
    contr.contract(2, 2);
    contr.contract(3, 3);
    btod_contract2<2, 2, 2>(contr, tp, tq).perform(tr_ref);

    letter i, j, k, l, a, b;
    invoke_eval(testname,
        contract(a|b, tp(i|j|a|b), tq(k|l|a|b)),
        i|j|k|l, tr_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_contr_2() throw(libtest::test_exception) {

    //
    //  Contraction of a tensor and a sum of two tensors, no symmetry
    //  r(i|j|k|l) = p(i|j|a|b) * [q1(k|l|a|b) + q2(k|l|a|b)]
    //

    static const char *testname = "direct_eval_test::test_contr_2()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si&sj|sa&sb), sijkl(si&sj&si&sj);

    btensor<4> tp(sijab), tq1(sijab), tq2(sijab);
    btod_random<4>().perform(tp);
    btod_random<4>().perform(tq1);
    btod_random<4>().perform(tq2);

    block_tensor<4, double, allocator_t> tr_ref(sijkl.get_bis());
    contraction2<2, 2, 2> contr;
    contr.contract(2, 2);
    contr.contract(3, 3);
    btod_contract2<2, 2, 2>(contr, tp, tq1).perform(tr_ref);
    btod_contract2<2, 2, 2>(contr, tp, tq2).perform(tr_ref, 1.0);

    letter i, j, k, l, a, b;
    invoke_eval(testname,
        contract(a|b, tp(i|j|a|b), tq1(k|l|a|b) + tq2(k|l|a|b)),
        i|j|k|l, tr_ref, 5e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_eval_test::test_mixed_1() throw(libtest::test_exception) {

    //
    //  Addition + contraction of two tensors, no symmetry
    //  s(i|j|k|l) = r(i|j|k|l) + p(i|j|a|b) * q(k|l|a|b)
    //

    static const char *testname = "direct_eval_test::test_mixed_1()";
    typedef std_allocator<double> allocator_t;

    try {

    bispace<1> si(5), sj(5), sa(10), sb(10);
    si.split(3);
    sj.split(3);
    sa.split(6);
    sb.split(6);
    bispace<4> sijab(si&sj|sa&sb), sijkl(si&sj&si&sj);

    btensor<4> tp(sijab), tq(sijab), tr(sijkl);
    btod_random<4>().perform(tp);
    btod_random<4>().perform(tq);
    btod_random<4>().perform(tr);

    block_tensor<4, double, allocator_t> ts_ref(sijkl.get_bis());
    contraction2<2, 2, 2> contr;
    contr.contract(2, 2);
    contr.contract(3, 3);
    btod_contract2<2, 2, 2>(contr, tp, tq).perform(ts_ref);
    btod_copy<4>(tr).perform(ts_ref, 1.0);

    letter i, j, k, l, a, b;
    invoke_eval(testname,
        tr(i|j|k|l) + contract(a|b, tp(i|j|a|b), tq(k|l|a|b)),
        i|j|k|l, ts_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
