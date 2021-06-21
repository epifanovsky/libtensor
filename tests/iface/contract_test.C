#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/libtensor.h>
#include "../compare_ref.h"
#include "contract_test.h"

namespace libtensor {


void contract_test::perform() {

    allocator<double>::init();

    try {

//        test_subexpr_labels_1();
//        test_contr_bld_1();
//        test_contr_bld_2();
        test_tt_1();
        test_tt_2();
        test_tt_3();
        test_tt_4();
        test_tt_5();
        test_tt_6();
        test_tt_7();
        test_tt_8();
        test_tt_9();
        test_tt_10();
        test_te_1();
        test_te_2();
        test_te_3();
        test_te_4();
        test_et_1();
        test_et_2();
        test_et_3();
        test_ee_1();
        test_ee_2();
        test_ee_3();
//        test_contract3_ttt_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}

#if 0
namespace {

using labeled_btensor_expr::expr_rhs;
using labeled_btensor_expr::contract2_core;
using labeled_btensor_expr::contract_subexpr_labels;

template<size_t N, size_t M, size_t NM, size_t K, typename T>
void test_subexpr_labels_tpl(
    expr_rhs<NM, T> e,
    letter_expr<NM> label_c) {

    const contract2_core<N, M, K, T> &core =
        dynamic_cast<const contract2_core<N, M, K, T>&>(e.get_core());
    contract_subexpr_labels<N, M, K, T> subexpr_labels(core, label_c);
}

} // unnamed namespace


void contract_test::test_subexpr_labels_1() {

    const char testname[] = "contract_test::test_subexpr_labels_1()";

    try {

    bispace<1> spi(4), spa(5);
    bispace<4> spijab((spi&spi)|(spa&spa));
    btensor<4> ta(spijab), tb(spijab);
    letter i, j, k, l, a, b;
    test_subexpr_labels_tpl<2, 2, 4, 2, double>(
        contract(a|b, ta(i|j|a|b), tb(k|l|a|b)),
        i|j|k|l);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_contr_bld_1() {

    const char testname[] = "contract_test::test_contr_bld_1()";

    try {

    letter i, j, k, l, a;
    letter_expr<4> label_c(i|j|k|l);
    letter_expr<1> label_contr(a);
    letter_expr<2> label_a(a|i);
    letter_expr<4> label_b(j|k|l|a);

    permutation<2> perm_a;
    permutation<4> perm_b;

    labeled_btensor_expr::contract_contraction2_builder<1, 3, 1> bld(
        label_a, perm_a, label_b, perm_b, label_c, label_contr);
    const contraction2<1, 3, 1> &contr = bld.get_contr();

    permutation<4> perm_c;
    contraction2<1, 3, 1> contr_ref(perm_c);
    contr_ref.contract(0, 3);

    for(size_t i = 0; i < 10; i++) {
        if(contr.get_conn().at(i) != contr_ref.get_conn().at(i)) {
            std::ostringstream ss;
            ss << "Incorrect connection at position " << i << ": "
                << contr.get_conn().at(i) << " vs. "
                << contr_ref.get_conn().at(i) << " (ref).";
            fail_test(testname, __FILE__, __LINE__,
                ss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_contr_bld_2() {

    const char testname[] = "contract_test::test_contr_bld_2()";

    try {

    letter i, j, k, l, a;
    letter_expr<4> label_c(i|j|k|l);
    letter_expr<1> label_contr(a);
    letter_expr<4> label_a(i|k|l|a);
    letter_expr<2> label_b(a|j);

    permutation<4> perm_a;
    permutation<2> perm_b;

    labeled_btensor_expr::contract_contraction2_builder<3, 1, 1> bld(
        label_a, perm_a, label_b, perm_b, label_c, label_contr);
    const contraction2<3, 1, 1> &contr = bld.get_contr();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr_ref(perm_c);
    contr_ref.contract(3, 0);

    for(size_t i = 0; i < 10; i++) {
        if(contr.get_conn().at(i) != contr_ref.get_conn().at(i)) {
            std::ostringstream ss;
            ss << "Incorrect connection at position " << i << ": "
                << contr.get_conn().at(i) << " vs. "
                << contr_ref.get_conn().at(i) << " (ref).";
            fail_test(testname, __FILE__, __LINE__,
                ss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}
#endif


void contract_test::test_tt_1() {

    const char testname[] = "contract_test::test_tt_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 3, 1> contr;
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(i|a), t2(b|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_2() {

    const char testname[] = "contract_test::test_tt_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(0, 1);
    contraction2<1, 3, 1> contr(perm_c);
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(i|b), t2(a|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_3() {

    const char testname[] = "contract_test::test_tt_3()";

    try {

    bispace<1> sp_i(10), sp_a(20), sp_b(19);
    bispace<2> sp_ib(sp_i|sp_b);
    bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_b|sp_a|sp_a);

    btensor<4> t1(sp_acdi);
    btensor<2> t2(sp_ib);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(a|c|d|i), t2(i|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_4() {

    const char testname[] = "contract_test::test_tt_4()";

    try {

    bispace<1> sp_i(10), sp_a(20), sp_b(19);
    bispace<2> sp_ib(sp_i|sp_b);
    bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_b|sp_a|sp_a);

    btensor<4> t1(sp_acdi);
    btensor<2> t2(sp_ib);
    btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref_tmp);
    btod_copy<4>(t3_ref_tmp, -1.0).perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = -contract(i, t1(a|c|d|i), t2(i|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_5() {

    const char testname[] = "contract_test::test_tt_5()";

    try {

    bispace<1> sp_i(10), sp_a(20), sp_b(19);
    bispace<2> sp_ib(sp_i|sp_b);
    bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_b|sp_a|sp_a);

    btensor<4> t1(sp_acdi);
    btensor<2> t2(sp_ib);
    btensor<4> t3(sp_abcd);
    btensor<4> t4(sp_abcd), t4_ref_tmp(sp_abcd), t4_ref(sp_abcd);

    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    btod_random<4>().perform(t3);
    t1.set_immutable();
    t2.set_immutable();
    t3.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t4_ref_tmp);
    btod_add<4> add(t3);
    add.add_op(t4_ref_tmp, -1.0);
    add.perform(t4_ref);

    letter a, b, c, d, i;
    t4(a|b|c|d) = t3(a|b|c|d) - contract(i, t1(a|c|d|i), t2(i|b));

    compare_ref<4>::compare(testname, t4, t4_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_6() {

    const char testname[] = "contract_test::test_tt_6()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a), sp_ab(sp_a|sp_a);
    bispace<4> sp_iabc(sp_i|sp_a|sp_a|sp_a);

    btensor<4> t1(sp_iabc);
    btensor<2> t2(sp_ia);
    btensor<2> t3(sp_ab), t3_ref(sp_ab);

    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<2, 0, 2> contr;
    contr.contract(0, 0);
    contr.contract(3, 1);
    btod_contract2<2, 0, 2>(contr, t1, t2).perform(t3_ref);

    letter a, b, c, i;
    t3(a|b) = contract(i|c, t1(i|a|b|c), t2(i|c));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_7() {

    const char testname[] = "contract_test::test_tt_7()";

    try {

    bispace<1> sp_i(13), sp_a(7);
    bispace<4> sp_ijab((sp_i&sp_i)|(sp_a&sp_a)), sp_iabc(sp_i|(sp_a&sp_a&sp_a));

    btensor<4> t1(sp_iabc);
    btensor<4> t2(sp_ijab);
    btensor<4> t3(sp_iabc), t3_ref(sp_iabc);

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    //  iabc = kcad ikbd
    //  caib->iabc
    contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2).
        permute(2, 3));
    contr.contract(0, 1);
    contr.contract(3, 3);

    btod_contract2<2, 2, 2>(contr, t1, t2).perform(t3_ref, 1.0);

    letter i, k, a, b, c, d;
    t3(i|a|b|c) = contract(k|d, t1(k|c|a|d), t2(i|k|b|d));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_8() {

    const char testname[] = "contract_test::test_tt_8()";

    try {

    bispace<1> sp_i(10), sp_a(3);
    sp_i.split(5);
    sp_a.split(2);
    bispace<2> sp_ab(sp_a&sp_a);
    bispace<4> sp_ijka((sp_i&sp_i&sp_i)|sp_a);

    btensor<4> t1(sp_ijka), t2(sp_ijka);
    btensor<2> t3(sp_ab), t3_ref(sp_ab);

    {
        block_tensor_ctrl<4, double> c1(t1), c2(t2);
        scalar_transf<double> tr1(-1.);
        c1.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        c2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
    }

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
    contr.contract(0, 0);
    contr.contract(1, 1);
    contr.contract(2, 2);

    btod_contract2<1, 1, 3>(contr, t1, t2).perform(t3_ref, -0.5);

    letter i, j, k, a, b;
        t3(a|b) =  0.5 * (-contract(i|j|k, t1(k|j|i|b), t2(k|j|i|a)));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_9() {

    const char testname[] = "contract_test::test_tt_9()";

    try {

    bispace<1> sp_i(10), sp_a(20), sp_k(11);
    sp_i.split(3).split(5);
    sp_a.split(6).split(13);
    bispace<4> sp_ijka((sp_i&sp_i)|sp_k|sp_a), sp_kija(sp_k|(sp_i&sp_i)|sp_a);
    bispace<4> sp_ijab((sp_i&sp_i)|(sp_a&sp_a));

    btensor<4> t1(sp_kija), t2(sp_ijab), t3(sp_ijka),
        t3_ref(sp_ijka);

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2));
    contr.contract(1, 1);
    contr.contract(3, 3);
    btod_contract2<2, 2, 2>(contr, t1, t2).perform(t3_ref);

    letter i, j, k, l, a, c;
    t3(i|j|k|a) = contract(l|c, t1(k|l|j|c), t2(i|l|a|c));

    {
        block_tensor_ctrl<4, double> c3(t3), c3_ref(t3_ref);
        symmetry<4, double> sym3(sp_ijka.get_bis()),
            sym3_ref(sp_ijka.get_bis());
        so_copy<4, double>(c3.req_const_symmetry()).perform(sym3);
        so_copy<4, double>(c3_ref.req_const_symmetry()).perform(sym3_ref);
        compare_ref<4>::compare(testname, sym3, sym3_ref);
    }

    compare_ref<4>::compare(testname, t3, t3_ref, 6e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_tt_10() {

    const char testname[] = "contract_test::test_tt_10()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 3, 1> contr;
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = -0.5 * contract(i, 2.0 * t1(i|a), -t2(b|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_te_1() {

    const char testname[] = "contract_test::test_te_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2a);
    btod_random<4>().perform(t2b);
    t1.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<4> op_add(t2a);
    op_add.add_op(t2b);
    op_add.perform(t2);
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(0, 1);
    contraction2<1, 3, 1> contr(perm_c);
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i) + t2b(a|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_te_2() {

    const char testname[] = "contract_test::test_te_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2a);
    btod_random<4>().perform(t2b);
    t1.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<4> op_add(t2a);
    op_add.add_op(t2b, -0.5);
    op_add.perform(t2);
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(0, 1);
    contraction2<1, 3, 1> contr(perm_c);
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i) - 0.5*t2b(a|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_te_3() {

    const char testname[] = "contract_test::test_te_3()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2a);
    btod_random<4>().perform(t2b);
    t1.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<4> op_add(t2a);
    op_add.add_op(t2b);
    op_add.perform(t2);
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(0, 1);
    contraction2<1, 3, 1> contr(perm_c);
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i)) +
        contract(i, t1(i|b), t2b(a|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_te_4() {

    const char testname[] = "contract_test::test_te_4()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2a);
    btod_random<4>().perform(t2b);
    t1.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<4> op_add(t2a);
    op_add.add_op(t2b, -0.5);
    op_add.perform(t2);
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(0, 1);
    contraction2<1, 3, 1> contr(perm_c);
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i)) -
        0.5*contract(i, t1(i|b), t2b(a|c|d|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_et_1() {

    const char testname[] = "contract_test::test_et_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
    btensor<2> t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    btod_random<2>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();
    t2.set_immutable();

    btod_add<4> op_add(t1a);
    op_add.add_op(t1b);
    op_add.perform(t1);
    t1.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1a(a|c|d|i) + t1b(a|c|d|i), t2(i|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_et_2() {

    const char testname[] = "contract_test::test_et_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
    btensor<2> t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    btod_random<2>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();
    t2.set_immutable();

    btod_add<4> op_add(t1a);
    op_add.add_op(t1b, 1.5);
    op_add.perform(t1);
    t1.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1a(a|c|d|i) + 1.5*t1b(a|c|d|i), t2(i|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_et_3() {

    const char testname[] = "contract_test::test_et_3()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<2> sp_ac(sp_a|sp_a), sp_di(sp_a|sp_i);
    bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1a(sp_ac), t1b(sp_di);
    btensor<4> t1(sp_acdi);
    btensor<2> t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1a);
    btod_random<2>().perform(t1b);
    btod_random<2>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();
    t2.set_immutable();

    contraction2<2, 2, 0> contr_t1;
    btod_contract2<2, 2, 0>(contr_t1, t1a, t1b).perform(t1);
    t1.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(i, t1a(a|c)*t1b(d|i), t2(i|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_ee_1() {

    const char testname[] = "contract_test::test_ee_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
    btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    btod_random<2>().perform(t2a);
    btod_random<2>().perform(t2b);
    t1a.set_immutable();
    t1b.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<4> op_add1(t1a);
    op_add1.add_op(t1b);
    op_add1.perform(t1);
    t1.set_immutable();

    btod_add<2> op_add2(t2a);
    op_add2.add_op(t2b);
    op_add2.perform(t2);
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(
        i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_ee_2() {

    const char testname[] = "contract_test::test_ee_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<2> sp_ai(sp_a|sp_i);
    bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_aicd(sp_a|sp_i|sp_a|sp_a);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<4> t1a(sp_acdi), t1b(sp_aicd), t1(sp_acdi);
    btensor<2> t2a(sp_ia), t2b(sp_ai), t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    btod_random<2>().perform(t2a);
    btod_random<2>().perform(t2b);
    t1a.set_immutable();
    t1b.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    permutation<4> perm_t1b;
    perm_t1b.permute(1, 2).permute(2, 3); // aicd->acid->acdi
    btod_add<4> op_add1(t1a);
    op_add1.add_op(t1b, perm_t1b);
    op_add1.perform(t1);
    t1.set_immutable();

    permutation<2> perm_t2b;
    perm_t2b.permute(0, 1); // bi->ib
    btod_add<2> op_add2(t2a);
    op_add2.add_op(t2b, perm_t2b);
    op_add2.perform(t2);
    t2.set_immutable();

    permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
    contraction2<3, 1, 1> contr(perm_c);
    contr.contract(3, 0);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = contract(
        i, t1a(a|c|d|i) + t1b(a|i|c|d), t2a(i|b) + t2b(b|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void contract_test::test_ee_3() {

    const char testname[] = "contract_test::test_ee_3()";
    const char pgname[] = "c2v";

    try {

    product_table_i::label_t a1 = 0, a2 = 1, b1 = 2, b2 = 3;
    std::vector<std::string> irreps(4);
    irreps[a1] = "A1"; irreps[a2] = "A2"; irreps[b1] = "B1"; irreps[b2] = "B2";
    point_group_table pg(pgname, irreps, irreps[a1]);
    pg.add_product(a1, a1, a1);
    pg.add_product(a1, a2, a2);
    pg.add_product(a1, b1, b1);
    pg.add_product(a1, b2, b2);
    pg.add_product(a2, a2, a1);
    pg.add_product(a2, b1, b2);
    pg.add_product(a2, b2, b1);
    pg.add_product(b1, b1, a1);
    pg.add_product(b1, b2, a2);
    pg.add_product(b2, b2, a1);

    product_table_container::get_instance().add(pg);

    } catch (exception &e) {
        product_table_container::get_instance().erase(pgname);
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    try {

    bispace<1> sp_x(16), sp_b(13);
    sp_x.split(4).split(7).split(8).split(12).split(15);
    bispace<2> sp_xx(sp_x&sp_x);
    bispace<2> sp_xb(sp_x|sp_b);
    bispace<2> sp_bb(sp_b&sp_b);

    btensor<2> t2a(sp_xx), t2b(sp_xb), t2c(sp_xb), t2(sp_bb), t2_ref(sp_bb);
    {
    const block_index_space<2> &bisa = t2a.get_bis(), &bisb = t2b.get_bis();
    dimensions<2> dimsa(bisa.get_block_index_dims()), dimsb(bisb.get_block_index_dims());

    block_tensor_wr_ctrl<2, double> ca(t2a), cb(t2b);
	symmetry<2, double> &sa = ca.req_symmetry(), &sb = cb.req_symmetry();

	scalar_transf<double> tr;
	se_perm<2, double> se(permutation<2>().permute(0, 1), tr);
	sa.insert(se);

	mask<2> ma, mb;
	ma[0] = ma[1] = mb[0] = true;
	index<2> i00, i01, i10, i11;
	i10[0] = i11[0] = i01[1] = i11[1] = 1;
	se_part<2, double> pa(bisa, ma, 2), pb(bisb, mb, 2);
	pa.add_map(i00, i11, tr);
	pa.mark_forbidden(i01);
	pa.mark_forbidden(i10);
	pb.add_map(i00, i10, tr);
	sa.insert(pa);
	sb.insert(pb);

	se_label<2, double> la(dimsa, pgname), lb(dimsb, pgname);
	block_labeling<2> &bla = la.get_labeling(), &blb = lb.get_labeling();
	bla.assign(ma, 0, 0); bla.assign(ma, 1, 2); bla.assign(ma, 2, 3);
	bla.assign(ma, 3, 0); bla.assign(ma, 4, 2); bla.assign(ma, 5, 3);
	blb.assign(mb, 0, 0); blb.assign(mb, 1, 2); blb.assign(mb, 2, 3);
	blb.assign(mb, 3, 0); blb.assign(mb, 4, 2); blb.assign(mb, 5, 3);
	la.set_rule(0);
	lb.set_rule(product_table_i::k_invalid);
	sa.insert(la);
	sb.insert(lb);

    }

    btod_random<2>().perform(t2a);
    btod_random<2>().perform(t2b);
    t2a.set_immutable();
    t2b.set_immutable();

    contraction2<1, 1, 1> c1, c2;
    c1.contract(1, 0);
    c2.contract(0, 0);

    btod_contract2<1, 1, 1> op1(c1, t2a, t2b);
    op1.perform(t2c);
    btod_contract2<1, 1, 1> op2(c2, t2b, t2c);
    op2.perform(t2_ref);
    btod_symmetrize2<2>(op2, 0, 1, true).perform(t2_ref);

    letter mu, nu, p, q;
    t2(mu|nu) = symm(mu, nu,
    		contract(p, t2b(p|mu), contract(q, t2a(p|q), t2b(q|nu))));

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-14);

    } catch(std::exception &e) {
        product_table_container::get_instance().erase(pgname);
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    product_table_container::get_instance().erase(pgname);
}


void contract_test::test_contract3_ttt_1() try {

    const char testname[] = "contract_test::test_contract3_ttt_1()";

    try {

    bispace<1> o(10), v(20);
    bispace<2> oo(o&o), ov(o|v);

    btensor<2> t1(ov), t2(ov), t3(ov), t4(ov), t4_ref(ov);
    btensor<2> tt(oo);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    btod_random<2>().perform(t3);
    t1.set_immutable();
    t2.set_immutable();
    t3.set_immutable();

    contraction2<1, 1, 1> contr1;
    contr1.contract(1, 1);
    btod_contract2<1, 1, 1>(contr1, t1, t2).perform(tt);
    contraction2<1, 1, 1> contr2;
    contr2.contract(1, 0);
    btod_contract2<1, 1, 1>(contr2, tt, t3).perform(t4_ref);

    letter a, b, i, j;
//    t4(i|a) = contract(b, t1(i|b), t2(j|b), j, t3(j|a));

    compare_ref<2>::compare(testname, t4, t4_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
} catch(...) { throw; }


} // namespace libtensor
