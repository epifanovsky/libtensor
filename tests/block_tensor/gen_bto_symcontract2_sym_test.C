#include <sstream>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_sym_impl.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_symcontract2_sym_impl.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/symmetry_element_set_adapter.h>
#include "../compare_ref.h"
#include "gen_bto_symcontract2_sym_test.h"

namespace libtensor {


void gen_bto_symcontract2_sym_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
}


void gen_bto_symcontract2_sym_test::test_1() {

    static const char testname[] = "gen_bto_symcontract2_sym_test::test_1()";

    //  c_klpq = P+(kl) a_kpr b_lqr
    //  No symmetry in a or b

    try {

        libtensor::index<3> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 5;
        dimensions<3> dimsa(index_range<3>(ia1, ia2));
        block_index_space<3> bisa(dimsa);
        mask<3> m110, m001;
        m110[0] = true; m110[1] = true; m001[2] = true;

        bisa.split(m110, 3);
        bisa.split(m110, 5);
        bisa.split(m001, 2);

        block_index_space<3> bisb(bisa);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 10; ic2[3] = 10;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bis_ref(dimsc);
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis_ref.split(m1111, 3);
        bis_ref.split(m1111, 5);

        symmetry<3, double> syma(bisa), symb(bisb);

        scalar_transf<double> tr0;

        symmetry<4, double> symc0_ref(bis_ref), symc_ref(bis_ref);
        permutation<4> p1023;
        p1023.permute(0, 1);
        se_perm<4, double> sp1023(p1023, tr0);
        symc_ref.insert(sp1023);

        // kplq -> klpq
        contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2));
        contr.contract(2, 2);

        gen_bto_symcontract2_sym<2, 2, 1, btod_traits> op(contr, syma, symb,
            p1023, true);

        compare_ref<4>::compare(testname, op.get_symmetry0(), symc0_ref);
        compare_ref<4>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void gen_bto_symcontract2_sym_test::test_2() {

    static const char testname[] = "gen_bto_symcontract2_sym_test::test_2()";

    //  c_klpq = P+(kp,lq) a_kpr b_lqr
    //  No symmetry in a or b

    try {

        libtensor::index<3> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 5;
        dimensions<3> dimsa(index_range<3>(ia1, ia2));
        block_index_space<3> bisa(dimsa);
        mask<3> m110, m001;
        m110[0] = true; m110[1] = true; m001[2] = true;

        bisa.split(m110, 3);
        bisa.split(m110, 5);
        bisa.split(m001, 2);

        block_index_space<3> bisb(bisa);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 10; ic2[3] = 10;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bis_ref(dimsc);
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis_ref.split(m1111, 3);
        bis_ref.split(m1111, 5);

        symmetry<3, double> syma(bisa), symb(bisb);

        scalar_transf<double> tr0;

        symmetry<4, double> symc0_ref(bis_ref), symc_ref(bis_ref);
        permutation<4> p1032;
        p1032.permute(0, 1).permute(2, 3);
        se_perm<4, double> sp1032(p1032, tr0);
        symc_ref.insert(sp1032);

        // kplq -> klpq
        contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2));
        contr.contract(2, 2);

        gen_bto_symcontract2_sym<2, 2, 1, btod_traits> op(contr, syma, symb,
            p1032, true);

        compare_ref<4>::compare(testname, op.get_symmetry0(), symc0_ref);
        compare_ref<4>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void gen_bto_symcontract2_sym_test::test_3() {

    static const char testname[] = "gen_bto_symcontract2_sym_test::test_3()";

    //  c_klpq = P+(kp,lq) a_kpr b_lqr
    //  Perm antisymmetry a_kpr = -a_pkr

    try {

        libtensor::index<3> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 5;
        dimensions<3> dimsa(index_range<3>(ia1, ia2));
        block_index_space<3> bisa(dimsa);
        mask<3> m110, m001;
        m110[0] = true; m110[1] = true; m001[2] = true;

        bisa.split(m110, 3);
        bisa.split(m110, 5);
        bisa.split(m001, 2);

        block_index_space<3> bisb(bisa);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 10; ic2[3] = 10;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bis_ref(dimsc);
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis_ref.split(m1111, 3);
        bis_ref.split(m1111, 5);

        symmetry<3, double> syma(bisa), symb(bisb);

        scalar_transf<double> tr0, tr1(-1.0);
        permutation<3> p102;
        p102.permute(0, 1);
        se_perm<3, double> sa102(p102, tr1);
        syma.insert(sa102);

        symmetry<4, double> symc0_ref(bis_ref), symc_ref(bis_ref);
        permutation<4> p1032, p2103;
        p1032.permute(0, 1).permute(2, 3);
        p2103.permute(0, 2);
        se_perm<4, double> sp1032(p1032, tr0);
        se_perm<4, double> sa2103(p2103, tr1);
        symc_ref.insert(sp1032);
        symc0_ref.insert(sa2103);

        // kplq -> klpq
        contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2));
        contr.contract(2, 2);

        gen_bto_symcontract2_sym<2, 2, 1, btod_traits> op(contr, syma, symb,
            p1032, true);

        compare_ref<4>::compare(testname, op.get_symmetry0(), symc0_ref);
        compare_ref<4>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

