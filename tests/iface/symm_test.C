#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/iface/iface.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "../compare_ref.h"
#include "symm_test.h"

namespace libtensor {


void symm_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_symm2_contr_tt_1();
        test_symm2_contr_ee_1();
        test_asymm2_contr_tt_1();
        test_asymm2_contr_tt_2();
        test_asymm2_contr_tt_3();
        test_asymm2_contr_tt_4();
        test_asymm2_contr_tt_5();
        test_asymm2_contr_tt_6();
        test_asymm2_contr_ee_1();
        test_asymm2_contr_ee_2();

        test_symm22_t_1();
        test_asymm22_t_1();
        test_symm22_t_2();
        test_asymm22_t_2();

        test_symm22_e_1();
        test_asymm22_e_1();
        test_symm22_e_2();
        test_asymm22_e_2();

        test_symm3_t_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void symm_test::test_symm2_contr_tt_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm2_contr_tt_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd), t3_ref_tmp(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 3, 1> contr;
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref_tmp);
    btod_copy<4> cp(t3_ref_tmp);
    btod_symmetrize2<4>(cp, permutation<4>().permute(0, 1), true).
        perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = symm(a, b, contract(i, t1(i|a), t2(b|c|d|i)));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_symm2_contr_ee_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm2_contr_ee_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
    btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

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
    op.perform(t3_ref_tmp);

    btod_copy<4> cp(t3_ref_tmp);
    btod_symmetrize2<4>(cp, permutation<4>().permute(0, 1), true).
        perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = symm(a, b,
        contract(i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b)));

    compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_tt_1() throw(libtest::test_exception) {

    const char *testname = "asym_contract_test::test_asymm2_contr_tt_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<4> t2(sp_bcdi);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd), t3_ref_tmp(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 3, 1> contr;
    contr.contract(0, 3);
    btod_contract2<1, 3, 1> op(contr, t1, t2);
    op.perform(t3_ref_tmp);
    btod_copy<4> cp(t3_ref_tmp);
    btod_symmetrize2<4>(cp, permutation<4>().permute(0, 1), false).
        perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = asymm(a, b, contract(i, t1(i|a), t2(b|c|d|i)));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_tt_2() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_tt_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ab(sp_a&sp_a);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t0(sp_ijab);
    btensor<4> t1(sp_ijab);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_ijab), t3_ref(sp_ijab), t3_ref_tmp(sp_ijab);

    btod_random<4>().perform(t0);
    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    t0.set_immutable();
    t1.set_immutable();
    t2.set_immutable();

    contraction2<3, 1, 1> contr;
    contr.contract(3, 1);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref_tmp);
    permutation<4> perm; perm.permute(2, 3);
    btod_copy<4>(t0).perform(t3_ref);
    btod_copy<4> cp(t3_ref_tmp);
    btod_symmetrize2<4>(cp, permutation<4>().permute(2, 3), false).
        perform(t3_ref, 1.0);

    letter i, j, a, b, c;
    t3(i|j|a|b) = t0(i|j|a|b) +
        asymm(a, b, contract(c, t1(i|j|a|c), t2(b|c)));

    compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_tt_3() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_tt_3()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ab(sp_a&sp_a);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1(sp_ijab);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_ijab), t3_ref(sp_ijab), t3_ref_tmp(sp_ijab);

    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<3, 1, 1> contr;
    contr.contract(3, 1);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref_tmp);
    btod_copy<4> cp(t3_ref_tmp, 1.5);
    btod_symmetrize2<4>(cp, permutation<4>().permute(2, 3), false).
        perform(t3_ref);

    letter i, j, a, b, c;
    t3(i|j|a|b) = 1.5 * asymm(a, b, contract(c, t1(i|j|a|c), t2(b|c)));

    compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_tt_4() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_tt_4()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ab(sp_a&sp_a);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1(sp_ijab);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_ijab), t3_ref(sp_ijab), t3_ref_tmp(sp_ijab);

    btod_random<4>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<3, 1, 1> contr;
    contr.contract(3, 1);
    btod_contract2<3, 1, 1> op(contr, t1, t2);
    op.perform(t3_ref_tmp);
    btod_copy<4> cp(t3_ref_tmp, 3.0);
    btod_symmetrize2<4>(cp, permutation<4>().permute(2, 3), false).
        perform(t3_ref);

    letter i, j, a, b, c;
    t3(i|j|a|b) = 1.5 * asymm(a, b,
        contract(c, t1(i|j|a|c), t2(b|c))) * 2.0;

    compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_tt_5() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_tt_5()";

    try {

    bispace<1> sp_i(10), sp_a(20), sp_k(11);
    sp_i.split(3).split(5);
    sp_a.split(6).split(13);
    bispace<4> sp_ijka(sp_i&sp_i|sp_k|sp_a), sp_kija(sp_k|sp_i&sp_i|sp_a);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1(sp_ijka), t2(sp_kija), t3(sp_ijab), t4(sp_ijka),
        t4_ref(sp_ijka), tt1(sp_ijka);

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<4>().perform(t3);
    t1.set_immutable();
    t2.set_immutable();
    t3.set_immutable();

    contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2));
    contr.contract(1, 1);
    contr.contract(3, 3);
    btod_contract2<2, 2, 2> op_contr(contr, t2, t3);
    btod_symmetrize2<4> op_symm(op_contr, 0, 1, false);
    op_symm.perform(tt1);
    btod_copy<4>(t1).perform(t4_ref);
    btod_copy<4>(tt1).perform(t4_ref, -1.0);

    letter i, j, k, l, a, b, c;
    t4(i|j|k|a) = t1(i|j|k|a) - asymm(i, j, contract(l|c,
        t2(k|l|j|c), t3(i|l|a|c)));

    {
        block_tensor_ctrl<4, double> c4(t4), c4_ref(t4_ref);
        symmetry<4, double> sym4(sp_ijka.get_bis()),
            sym4_ref(sp_ijka.get_bis());
        so_copy<4, double>(c4.req_const_symmetry()).perform(sym4);
        so_copy<4, double>(c4_ref.req_const_symmetry()).perform(sym4_ref);
        compare_ref<4>::compare(testname, sym4, sym4_ref);
    }

    compare_ref<4>::compare(testname, t4, t4_ref, 6e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_tt_6() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_tt_6()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    sp_i.split(3).split(5).split(8);
    sp_a.split(6).split(10).split(16);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<2, double> t1(sp_ia);
    btensor<4, double> t2(sp_ijab), t3(sp_ijab), t3_ref(sp_ijab);

    {
        block_tensor_ctrl<2, double> ctrl(t1);

        mask<2> m11;
        m11[0] = true; m11[1] = true;

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;

        se_part<2, double> se(sp_ia.get_bis(), m11, 2);
        se.add_map(i00, i11);
        se.mark_forbidden(i01);
        se.mark_forbidden(i10);
        ctrl.req_symmetry().insert(se);
    }
    {
        block_tensor_ctrl<4, double> ctrl(t2);

        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
            i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;

        se_part<4,double> se(sp_ijab.get_bis(), m1111, 2);
        se.add_map(i0000, i1111);
        se.add_map(i0110, i1001);
        se.add_map(i0101, i1010);
        se.mark_forbidden(i0001);
        se.mark_forbidden(i0010);
        se.mark_forbidden(i0100);
        se.mark_forbidden(i1000);
        se.mark_forbidden(i0011);
        se.mark_forbidden(i1100);
        se.mark_forbidden(i0111);
        se.mark_forbidden(i1011);
        se.mark_forbidden(i1101);
        se.mark_forbidden(i1110);
        ctrl.req_symmetry().insert(se);
    }

    btod_random<2>().perform(t1);
    btod_random<4>().perform(t2);
    btod_random<4>().perform(t3);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<2, 2, 0> contr(permutation<4>().permute(1, 2));
    btod_contract2<2, 2, 0> op_contr(contr, t1, t1);
    btod_symmetrize2<4> op_symm(op_contr, 0, 1, false);
    btod_copy<4>(t2).perform(t3_ref);
    op_symm.perform(t3_ref, 1.0);

    letter i, j, k, l, a, b, c;
    t3(i|j|a|b) = t2(i|j|a|b) + asymm(i, j, t1(i|a) * t1(j|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void symm_test::test_asymm2_contr_ee_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_ee_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
    btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
    btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

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
    op.perform(t3_ref_tmp);

    btod_copy<4> cp(t3_ref_tmp);
    btod_symmetrize2<4>(cp, permutation<4>().permute(0, 1), false).
        perform(t3_ref);

    letter a, b, c, d, i;
    t3(a|b|c|d) = asymm(a, b,
        contract(i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b)));

    compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void symm_test::test_asymm2_contr_ee_2() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm2_contr_ee_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    sp_i.split(5);
    sp_a.split(8); sp_a.split(16);
    bispace<2> sp_ij(sp_i|sp_i);
    bispace<4> sp_ijab(sp_i|sp_i|sp_a|sp_a);

    btensor<2> bta(sp_ij);
    btensor<4> btb(sp_ijab), btc(sp_ijab), btc_ref(sp_ijab);

    permutation<2> p10;
    p10.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<2, double> sp10(p10, tr0);
    permutation<4> p1023, p0132;
    p1023.permute(0, 1); p0132.permute(2, 3);
    se_perm<4, double> sp1023(p1023, tr1), sp0132(p0132, tr1);

    btod_random<2>().perform(bta);
    btod_random<4>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    contraction2<1, 3, 1> contr;
    contr.contract(1, 1);
    btod_contract2<1, 3, 1> op(contr, bta, btb);
    btod_symmetrize2<4> op_sym(op, 0, 1, false);
    op_sym.perform(btc_ref);

    letter a, b, i, j, k;
    btc(i|j|a|b) = asymm(i, j, contract(k, bta(i|k), btb(j|k|a|b)));

    compare_ref<4>::compare(testname, btc, btc_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}



/** \test Tests the symmetrization over two pairs of indexes P+(ij)P+(ab)
        in a %tensor
 **/
void symm_test::test_symm22_t_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm22_t_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1(sp_ijab), t2(sp_ijab), t2_ref(sp_ijab),
        t2_ref_tmp(sp_ijab);

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();

    btod_copy<4> cp1(t1);
    btod_symmetrize2<4>(cp1, permutation<4>().permute(0, 1), true).
        perform(t2_ref_tmp);
    btod_copy<4> cp2(t2_ref_tmp);
    btod_symmetrize2<4>(cp2, permutation<4>().permute(2, 3), true).
        perform(t2_ref);

    letter i, j, a, b;
    t2(i|j|a|b) = symm(i, j, symm(a, b, t1(i|j|a|b)));

    compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the anti-symmetrization over two pairs of indexes
        P-(ij)P-(ab) in a %tensor
 **/
void symm_test::test_asymm22_t_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm22_t_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1(sp_ijab), t2(sp_ijab), t2_ref(sp_ijab),
        t2_ref_tmp(sp_ijab);

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();

    btod_copy<4> cp1(t1);
    btod_symmetrize2<4>(cp1, permutation<4>().permute(0, 1), false).
        perform(t2_ref_tmp);
    btod_copy<4> cp2(t2_ref_tmp);
    btod_symmetrize2<4>(cp2, permutation<4>().permute(2, 3), false).
        perform(t2_ref);

    letter i, j, a, b;
    t2(i|j|a|b) = asymm(i, j, asymm(a, b, t1(i|j|a|b)));

    compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the symmetrization over two pairs of indexes P+(i|jk)
        in a %tensor
 **/
void symm_test::test_symm22_t_2() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm22_t_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

    btensor<4> t1(sp_ijka), t2a(sp_ijka), t2b(sp_ijka), t2_ref(sp_ijka);

    btod_random<4>().perform(t1);
    t1.set_immutable();

    permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
    permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
    btod_copy<4>(t1).perform(t2_ref);
    btod_copy<4>(t1, perm1).perform(t2_ref, 1.0);
    btod_copy<4>(t1, perm2).perform(t2_ref, 1.0);

    letter i, j, k, a;
    t2a(i|j|k|a) = symm(i, j, k, t1(i|j|k|a));
    t2b(i|j|k|a) = symm(i, k, j, t1(i|j|k|a));

    compare_ref<4>::compare(testname, t2a, t2b, 1e-15);
//    compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the anti-symmetrization over two pairs of indexes P-(i|jk)
        in a %tensor
 **/
void symm_test::test_asymm22_t_2() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm22_t_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

    btensor<4> t1(sp_ijka), t2a(sp_ijka), t2b(sp_ijka), t2_ref(sp_ijka);

    btod_random<4>().perform(t1);
    t1.set_immutable();

    permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
    permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
    btod_copy<4>(t1).perform(t2_ref);
    btod_copy<4>(t1, perm1).perform(t2_ref, -1.0);
    btod_copy<4>(t1, perm2).perform(t2_ref, -1.0);

    letter i, j, k, a;
    t2a(i|j|k|a) = asymm(i, j, k, t1(i|j|k|a));
    t2b(i|j|k|a) = asymm(i, k, j, t1(i|j|k|a));

    compare_ref<4>::compare(testname, t2a, t2b, 1e-15);
    //compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the symmetrization over two pairs of indexes P+(ij)P+(ab)
        in an expression
 **/
void symm_test::test_symm22_e_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm22_e_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1a(sp_ijab), t1b(sp_ijab), t1(sp_ijab),
        t2(sp_ijab), t2_ref(sp_ijab), t2_ref_tmp(sp_ijab);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    btod_random<4>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();

    btod_copy<4>(t1a).perform(t1);
    btod_copy<4>(t1b).perform(t1, 1.0);
    btod_copy<4> cp1(t1);
    btod_symmetrize2<4>(cp1, permutation<4>().permute(0, 1), true).
        perform(t2_ref_tmp);
    btod_copy<4> cp2(t2_ref_tmp);
    btod_symmetrize2<4>(cp2, permutation<4>().permute(2, 3), true).
        perform(t2_ref);

    letter i, j, a, b;
    t2(i|j|a|b) = symm(i, j, symm(a, b, t1a(i|j|a|b) + t1b(i|j|a|b)));

    compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the anti-symmetrization over two pairs of indexes
        P-(ij)P-(ab) in an expression
 **/
void symm_test::test_asymm22_e_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm22_e_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

    btensor<4> t1a(sp_ijab), t1b(sp_ijab), t1(sp_ijab),
        t2(sp_ijab), t2_ref(sp_ijab), t2_ref_tmp(sp_ijab);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    btod_random<4>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();

    btod_copy<4>(t1a).perform(t1);
    btod_copy<4>(t1b).perform(t1, 1.0);
    btod_copy<4> cp1(t1);
    btod_symmetrize2<4>(cp1, permutation<4>().permute(0, 1), false).
        perform(t2_ref_tmp);
    btod_copy<4> cp2(t2_ref_tmp);
    btod_symmetrize2<4>(cp2, permutation<4>().permute(2, 3), false).
        perform(t2_ref);

    letter i, j, a, b;
    t2(i|j|a|b) = asymm(i, j, asymm(a, b, t1a(i|j|a|b) + t1b(i|j|a|b)));

    compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the symmetrization over two pairs of indexes P+(i|jk)
        in an expression
 **/
void symm_test::test_symm22_e_2() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm22_e_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

    btensor<4> t1a(sp_ijka), t1b(sp_ijka), t1(sp_ijka), t2a(sp_ijka),
        t2b(sp_ijka), t2_ref(sp_ijka);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    t1a.set_immutable();
    t1b.set_immutable();

    permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
    permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
    btod_copy<4>(t1a).perform(t1);
    btod_copy<4>(t1b).perform(t1, 1.0);
    btod_copy<4>(t1).perform(t2_ref);
    btod_copy<4>(t1, perm1).perform(t2_ref, 1.0);
    btod_copy<4>(t1, perm2).perform(t2_ref, 1.0);

    letter i, j, k, a;
    t2a(i|j|k|a) = symm(i, j, k, t1a(i|j|k|a) + t1b(i|j|k|a));
    t2b(i|j|k|a) = symm(i, k, j, t1a(i|j|k|a) + t1b(i|j|k|a));

    compare_ref<4>::compare(testname, t2a, t2b, 1e-15);
    //compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the anti-symmetrization over two pairs of indexes P-(i|jk)
        in an expression
 **/
void symm_test::test_asymm22_e_2() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_asymm22_e_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

    btensor<4> t1a(sp_ijka), t1b(sp_ijka), t1(sp_ijka), t2a(sp_ijka),
        t2b(sp_ijka), t2_ref(sp_ijka);

    btod_random<4>().perform(t1a);
    btod_random<4>().perform(t1b);
    t1a.set_immutable();
    t1b.set_immutable();

    permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
    permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
    btod_copy<4>(t1a).perform(t1);
    btod_copy<4>(t1b).perform(t1, 1.0);
    btod_copy<4>(t1).perform(t2_ref);
    btod_copy<4>(t1, perm1).perform(t2_ref, -1.0);
    btod_copy<4>(t1, perm2).perform(t2_ref, -1.0);

    letter i, j, k, a;
    t2a(i|j|k|a) = asymm(i, j, k, t1a(i|j|k|a) + t1b(i|j|k|a));
    t2b(i|j|k|a) = asymm(i, k, j, t1a(i|j|k|a) + t1b(i|j|k|a));

    compare_ref<4>::compare(testname, t2a, t2b, 1e-15);
//    compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the symmetrization over three indexes in a %tensor
 **/
void symm_test::test_symm3_t_1() throw(libtest::test_exception) {

    const char *testname = "symm_test::test_symm3_t_1()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t1(sp_ijk), t2(sp_ijk), t2_ref(sp_ijk);

    btod_random<3>().perform(t1);
    btod_random<3>().perform(t2);
    t1.set_immutable();

    btod_copy<3>(t1).perform(t2_ref);
    btod_copy<3>(t1, permutation<3>().permute(0, 1)).perform(t2_ref, 1.0);
    btod_copy<3>(t1, permutation<3>().permute(0, 2)).perform(t2_ref, 1.0);
    btod_copy<3>(t1, permutation<3>().permute(1, 2)).perform(t2_ref, 1.0);
    btod_copy<3>(t1, permutation<3>().permute(0, 1).permute(0, 2)).
        perform(t2_ref, 1.0);
    btod_copy<3>(t1, permutation<3>().permute(0, 1).permute(1, 2)).
        perform(t2_ref, 1.0);

    letter i, j, k;
    t2(i|j|k) = symm(i, j, k, t1(i|j|k));

    typedef std_allocator<double> allocator_t;
    dense_tensor<3, double, allocator_t> tt2(sp_ijk.get_bis().get_dims()),
        tt2_ref(sp_ijk.get_bis().get_dims());
    tod_btconv<3>(t2).perform(tt2);
    tod_btconv<3>(t2_ref).perform(tt2_ref);

    compare_ref<3>::compare(testname, tt2, tt2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
