#include <sstream>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_dirsum_sym_impl.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/symmetry_element_set_adapter.h>
#include "../compare_ref.h"
#include "gen_bto_dirsum_sym_test.h"

namespace libtensor {


void gen_bto_dirsum_sym_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
}


void gen_bto_dirsum_sym_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "gen_bto_dirsum_sym_test::test_1()";

    try {

        index<2> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        block_index_space<2> bisa(dimsa);
        mask<2> ma;
        ma[0] = true; ma[1] = true;

        bisa.split(ma, 3);
        bisa.split(ma, 5);

        block_index_space<2> bisb(bisa);

        index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 10; ic2[3] = 10;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bis_ref(dimsc);
        mask<4> msk;
        msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
        bis_ref.split(msk, 3);
        bis_ref.split(msk, 5);

        symmetry<2, double> syma(bisa), symb(bisb);

        permutation<2> p10;
        p10.permute(0, 1);
        scalar_transf<double> tr0;
        se_perm<2, double> sp10(p10, tr0);
        syma.insert(sp10);
        symb.insert(sp10);

        symmetry<4, double> symc_ref(bis_ref);
        permutation<4> p1023, p0132;
        p1023.permute(0, 1);
        p0132.permute(2, 3);
        se_perm<4, double> sp1023(p1023, tr0), sp0132(p0132, tr0);
        symc_ref.insert(sp1023);
        symc_ref.insert(sp0132);

        gen_bto_dirsum_sym<2, 2, btod_traits> op(syma, tr0,
                symb, tr0, permutation<4>(), false);

        compare_ref<4>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void gen_bto_dirsum_sym_test::test_2() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ij + b_kl
    //  Permutational anti-symmetry in [ij] and [kl]
    //

    static const char *testname = "gen_bto_dirsum_sym_test::test_2()";

    try {

        index<2> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        block_index_space<2> bisa(dimsa);
        mask<2> ma;
        ma[0] = true; ma[1] = true;

        bisa.split(ma, 3);
        bisa.split(ma, 5);

        block_index_space<2> bisb(bisa);

        index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 10; ic2[3] = 10;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bis_ref(dimsc);
        mask<4> msk;
        msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
        bis_ref.split(msk, 3);
        bis_ref.split(msk, 5);

        symmetry<2, double> syma(bisa), symb(bisb);

        permutation<2> p10;
        p10.permute(0, 1);
        scalar_transf<double> tr0, tr1(-1.0);
        se_perm<2, double> sp10(p10, tr1);
        syma.insert(sp10);
        symb.insert(sp10);

        symmetry<4, double> symc_ref(bis_ref);
        permutation<4> p1032;
        p1032.permute(0, 1).permute(2, 3);
        se_perm<4, double> sp1032(p1032, tr1);
        symc_ref.insert(sp1032);

        gen_bto_dirsum_sym<2, 2, btod_traits> op(syma, tr0,
                symb, tr0, permutation<4>(), false);

        compare_ref<4>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void gen_bto_dirsum_sym_test::test_3() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijk + b_l
    //  Permutational symmetry in (ijk) and anti-symmetry in (ij)
    //

    static const char *testname = "gen_bto_dirsum_sym_test::test_3()";

    try {

        index<3> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 10;
        dimensions<3> dimsa(index_range<3>(ia1, ia2));
        index<1> ib1, ib2;
        ib2[0] = 8;
        dimensions<1> dimsb(index_range<1>(ib1, ib2));
        index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 10; ic2[3] = 8;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));

        block_index_space<3> bisa(dimsa);
        block_index_space<1> bisb(dimsb);
        block_index_space<4> bisc(dimsc);

        mask<3> ma;
        ma[0] = true; ma[1] = true; ma[2] = true;
        mask<1> mb;
        mb[0] = true;
        mask<4> mc1, mc2;
        mc1[0] = true; mc1[1] = true; mc1[2] = true; mc2[3] = true;

        bisa.split(ma, 3);
        bisa.split(ma, 5);
        bisb.split(mb, 4);
        bisc.split(mc1, 3);
        bisc.split(mc1, 5);
        bisc.split(mc2, 4);

        symmetry<3, double> syma(bisa);
        symmetry<1, double> symb(bisb);
        symmetry<4, double> symc_ref(bisc);

        permutation<3> p120, p102;
        p120.permute(0, 1).permute(1, 2);
        p102.permute(0, 1);

        permutation<4> p1203;
        p1203.permute(0, 1).permute(1, 2);

        scalar_transf<double> tr0, tr1(-1.0);
        se_perm<3, double> cycle3a(p120, tr0), cycle3b(p102, tr1);
        se_perm<4, double> cycle4(p1203, tr0);

        syma.insert(cycle3a);
        syma.insert(cycle3b);
        symc_ref.insert(cycle4);

        gen_bto_dirsum_sym<3, 1, btod_traits> op(syma, tr0,
                symb, tr1, permutation<4>());

        compare_ref<4>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void gen_bto_dirsum_sym_test::test_4() throw(libtest::test_exception) {

    //
    //  c_ij = a_i - a_j
    //  Permutational anti-symmetry in (i-j)
    //

    static const char *testname = "gen_bto_dirsum_sym_test::test_4()";

    try {

        index<1> ia1, ia2;
        ia2[0] = 8;
        dimensions<1> dimsa(index_range<1>(ia1, ia2));

        index<2> ic1, ic2;
        ic2[0] = 8; ic2[1] = 8;
        dimensions<2> dimsc(index_range<2>(ic1, ic2));

        block_index_space<1> bisa(dimsa);
        block_index_space<2> bisc(dimsc);

        mask<1> ma;
        ma[0] = true;
        mask<2> mc;
        mc[0] = true; mc[1] = true;

        bisa.split(ma, 3);
        bisa.split(ma, 5);
        bisc.split(mc, 3);
        bisc.split(mc, 5);

        symmetry<1, double> syma(bisa);
        symmetry<2, double> symc_ref(bisc);

        permutation<2> p10;
        p10.permute(0, 1);

        scalar_transf<double> tr0, tr1(-1.0);
        se_perm<2, double> cycle2(p10, tr1);

        symc_ref.insert(cycle2);

        gen_bto_dirsum_sym<1, 1, btod_traits> op(syma, tr0,
                syma, tr1, permutation<2>(), true);

        compare_ref<2>::compare(testname, op.get_symmetry(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

