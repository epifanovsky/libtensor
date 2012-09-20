#include <sstream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/symmetry_element_set_adapter.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_sym_impl.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include "../compare_ref.h"
#include "bto_contract2_sym_test.h"

namespace libtensor {


void bto_contract2_sym_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
}


void bto_contract2_sym_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "bto_contract2_sym_test::test_1()";

    try {

        index<4> i1, i2;
        i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bisa(dims), bis_ref(dims);
        mask<4> msk;
        msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;

        bisa.split(msk, 3);
        bisa.split(msk, 5);
        bis_ref.split(msk, 3);
        bis_ref.split(msk, 5);

        block_index_space<4> bisb(bisa);
        dimensions<4> bidimsa(bisa.get_block_index_dims()),
            bidimsb(bisb.get_block_index_dims()),
            bidimsc(bis_ref.get_block_index_dims());

        symmetry<4, double> syma(bisa), symb(bisb);

        permutation<4> p1230, p1023, p0132;
        p1230.permute(0, 1).permute(1, 2).permute(2, 3);
        p1023.permute(0, 1);
        p0132.permute(2, 3);
        scalar_transf<double> tr0;
        se_perm<4, double> cycle4a(p1230, tr0), cycle2a(p1023, tr0);
        syma.insert(cycle4a);
        syma.insert(cycle2a);
        symb.insert(cycle4a);
        symb.insert(cycle2a);

        symmetry<4, double> symc_ref(bis_ref);
        se_perm<4, double> cycle2c_1(p1023, tr0), cycle2c_2(p0132, tr0);
        symc_ref.insert(cycle2c_1);
        symc_ref.insert(cycle2c_2);

        contraction2<2, 2, 2> contr;
        contr.contract(0, 2);
        contr.contract(1, 3);

        gen_bto_contract2_sym<2, 2, 2, btod_traits> op(contr, bisa, syma, bisb,
            symb);

        compare_ref<4>::compare(testname, op.get_symc(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_sym_test::test_2() throw(libtest::test_exception) {

    //
    //  c_ijk = a_ipqr b_jpqrk
    //  Permutational symmetry in ijpqr
    //

    static const char *testname = "bto_contract2_sym_test::test_2()";

    try {

        index<3> i3_1, i3_2;
        i3_2[0] = 10; i3_2[1] = 10; i3_2[2] = 8;
        dimensions<3> dims3(index_range<3>(i3_1, i3_2));
        index<4> i4_1, i4_2;
        i4_2[0] = 10; i4_2[1] = 10; i4_2[2] = 10; i4_2[3] = 10;
        dimensions<4> dims4(index_range<4>(i4_1, i4_2));
        index<5> i5_1, i5_2;
        i5_2[0] = 10; i5_2[1] = 10; i5_2[2] = 10; i5_2[3] = 10; i5_2[4] = 8;
        dimensions<5> dims5(index_range<5>(i5_1, i5_2));

        block_index_space<4> bisa(dims4);
        block_index_space<5> bisb(dims5);
        block_index_space<3> bis_ref(dims3);

        mask<3> msk3_1, msk3_2;
        msk3_1[0] = true; msk3_1[1] = true; msk3_2[2] = true;
        mask<4> msk4;
        msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
        mask<5> msk5_1, msk5_2;
        msk5_1[0] = true; msk5_1[1] = true; msk5_1[2] = true; msk5_1[3] = true;
        msk5_2[4] = true;

        bisa.split(msk4, 3);
        bisa.split(msk4, 5);
        bisb.split(msk5_1, 3);
        bisb.split(msk5_1, 5);
        bisb.split(msk5_2, 4);
        bis_ref.split(msk3_1, 3);
        bis_ref.split(msk3_1, 5);
        bis_ref.split(msk3_2, 4);

        symmetry<4, double> syma(bisa);
        symmetry<5, double> symb(bisb);

        permutation<4> p1230, p1023;
        p1230.permute(0, 1).permute(1, 2).permute(2, 3);
        p1023.permute(0, 1);

        permutation<5> p12304, p10234;
        p12304.permute(0, 1).permute(1, 2).permute(2, 3);
        p10234.permute(0, 1);

        symmetry<3, double> symc_ref(bis_ref);
        scalar_transf<double> tr0;
        se_perm<4, double> cycle4a_1(p1230, tr0), cycle4a_2(p1023, tr0);
        se_perm<5, double> cycle4b_1(p12304, tr0), cycle4b_2(p10234, tr0);

        syma.insert(cycle4a_1);
        syma.insert(cycle4a_2);
        symb.insert(cycle4b_1);
        symb.insert(cycle4b_2);

        contraction2<1, 2, 3> contr;
        contr.contract(1, 1);
        contr.contract(2, 2);
        contr.contract(3, 3);

        gen_bto_contract2_sym<1, 2, 3, btod_traits> op(contr, bisa, syma, bisb,
            symb);

        compare_ref<3>::compare(testname, op.get_symc(), symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_sym_test::test_3() throw(libtest::test_exception) {

    //
    //  c_ijpq = a_ijpr b_qr
    //  Permutational symmetry in (i-j) (p-r) (q-r)
    //

    static const char *testname = "bto_contract2_sym_test::test_3()";

    try {

        index<2> i2_1, i2_2;
        i2_2[0] = 10; i2_2[1] = 10;
        dimensions<2> dims2(index_range<2>(i2_1, i2_2));
        index<4> i4_1, i4_2;
        i4_2[0] = 8; i4_2[1] = 8; i4_2[2] = 10; i4_2[3] = 10;
        dimensions<4> dims4(index_range<4>(i4_1, i4_2));

        block_index_space<4> bisa(dims4);
        block_index_space<2> bisb(dims2);

        mask<2> msk2;
        msk2[0] = true; msk2[1] = true;
        mask<4> msk4_1, msk4_2;
        msk4_1[0] = true; msk4_1[1] = true; msk4_2[2] = true; msk4_2[3] = true;

        bisa.split(msk4_1, 3);
        bisa.split(msk4_1, 5);
        bisa.split(msk4_2, 4);
        bisb.split(msk2, 4);

        symmetry<4, double> syma(bisa);
        symmetry<2, double> symb(bisb);

        permutation<4> p0132, p1023;
        p0132.permute(2, 3);
        p1023.permute(0, 1);

        permutation<2> p10;
        p10.permute(0, 1);

        scalar_transf<double> tr0, tr1(-1.0);
        se_perm<2, double> cycle2(p10, tr0);
        se_perm<4, double> cycle4a(p1023, tr1), cycle4b(p0132, tr1);

        syma.insert(cycle4a);
        syma.insert(cycle4b);
        symb.insert(cycle2);

        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        gen_bto_contract2_sym<3, 1, 1, btod_traits> op(contr, bisa, syma, bisb,
            symb);

        const symmetry<4, double> &sym = op.get_symc();
        symmetry<4, double>::iterator is = sym.begin();
        const symmetry_element_set<4, double> &set = sym.get_subset(is);
        symmetry_element_set_adapter<4, double, se_perm<4, double> > adapter(set);
        permutation_group<4, double> grp(set);
        if (! grp.is_member(tr1, p1023)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Permutational anti-symmetry (0-1) missing.");
        }
        if (grp.is_member(tr1, p0132)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Bad permutational anti-symmetry (2-3).");
        }

        //~ if(!op.get_symmetry().equals(sym_ref)) {
        //~ fail_test(testname, __FILE__, __LINE__,
        //~ "Symmetry does not match reference.");
        //~ }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_sym_test::test_4() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_klab b_klab
    //  Permutational symmetry in (i-j) (a-b) (k-l)
    //

    static const char *testname = "bto_contract2_sym_test::test_4()";

    try {

        index<4> i4_1, i4_2;
        i4_2[0] = 8; i4_2[1] = 8; i4_2[2] = 10; i4_2[3] = 10;
        dimensions<4> dims4(index_range<4>(i4_1, i4_2));

        block_index_space<4> bis(dims4);

        mask<4> msk4_1, msk4_2;
        msk4_1[0] = true; msk4_1[1] = true; msk4_2[2] = true; msk4_2[3] = true;

        bis.split(msk4_1, 4);
        bis.split(msk4_2, 3);
        bis.split(msk4_2, 5);

        symmetry<4, double> syma(bis), symb(bis);

        permutation<4> p0132, p1023, p2301;
        p0132.permute(2, 3);
        p1023.permute(0, 1);
        p2301.permute(0, 2).permute(1, 3);

        scalar_transf<double> tr1(-1.0);
        se_perm<4, double> cycle4a(p1023, tr1), cycle4b(p0132, tr1);

        syma.insert(cycle4a);
        syma.insert(cycle4b);
        symb.insert(cycle4a);
        symb.insert(cycle4b);

        contraction2<2, 2, 2> contr(p2301);
        contr.contract(2, 2);
        contr.contract(3, 3);

        gen_bto_contract2_sym<2, 2, 2, btod_traits> op(contr, bis, syma, bis,
            symb);

        const symmetry<4, double> &sym = op.get_symc();
        symmetry<4, double>::iterator is = sym.begin();
        const symmetry_element_set<4, double> &set = sym.get_subset(is);
        symmetry_element_set_adapter<4, double, se_perm<4, double> > adapter(set);
        permutation_group<4, double> grp(set);
        if (! grp.is_member(tr1, p1023)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Permutational anti-symmetry (0-1) missing.");
        }
        if (! grp.is_member(tr1, p0132)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Permutational anti-symmetry (2-3) missing.");
        }

        //~ if(!op.get_symmetry().equals(sym_ref)) {
        //~ fail_test(testname, __FILE__, __LINE__,
        //~ "Symmetry does not match reference.");
        //~ }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

