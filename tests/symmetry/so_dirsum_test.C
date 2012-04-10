#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_dirsum.h>
#include "../compare_ref.h"
#include "so_dirsum_test.h"
#include "../../../adcman2/supplement/supplementary_print.h"

namespace libtensor {


void so_dirsum_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2();
    test_empty_3();
    test_se_1( true, true); test_se_1( true, false);
    test_se_1(false, true); test_se_1(false, false);
    test_se_2( true, true); test_se_2( true, false);
    test_se_2(false, true); test_se_2(false, false);
    test_se_3();
    test_se_4();
    test_perm_1();
    test_perm_2();
    test_vac_1();
    test_vac_2();
}


/** \test Direct product of empty symmetry in 2-space and empty symmetry in
        1-space to form a 3-space. Expects empty symmetry in 3-space.
 **/
void so_dirsum_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_empty_1()";

    try {

    index<2> i2a, i2b; i2b[0] = 5; i2b[1] = 5;
    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    index<1> i1a, i1b; i1b[0] = 10;
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    index<3> i3a, i3b; i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
    dimensions<3> dimsc(index_range<3>(i3a, i3b));

    block_index_space<2> bisa(dimsa);
    block_index_space<1> bisb(dimsb);
    block_index_space<3> bisc(dimsc);

    mask<2> ma; ma[0] = true; ma[1] = true;
    bisa.split(ma, 2); bisa.split(ma, 3);
    mask<1> mb; mb[0] = true;
    bisb.split(mb, 5);
    mask<3> mc1, mc2; mc1[0] = true; mc1[1] = true; mc2[2] = true;
    bisc.split(mc1, 2); bisc.split(mc1, 3); bisc.split(mc2, 5);

    symmetry<2, double> syma(bisa);
    symmetry<1, double> symb(bisb);
    symmetry<3, double> symc(bisc);
    symmetry<3, double> symc_ref(bisc);
    so_dirsum<2, 1, double>(syma, symb).perform(symc);

    symmetry<3, double>::iterator i = symc.begin();
    if(i != symc.end()) {
        fail_test(testname, __FILE__, __LINE__, "i != symc.end()");
    }

    compare_ref<3>::compare(testname, symc, symc_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of non-empty symmetry in 2-space and empty symmetry
        in 1-space to form a 3-space. Expects non-empty symmetry in 3-space.
 **/
void so_dirsum_test::test_empty_2() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_empty_2()";

}

/** \test Direct product of empty symmetry in 1-space and non-empty symmetry in
        2-space to form a 3-space. Expects non-empty symmetry in 3-space.
 **/
void so_dirsum_test::test_empty_3() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_empty_3()";

}

/** \test Direct product of two S2 symmetries in 2-space forming a 4-space.
        Expects S2*S2 in 4-space.
 **/
void so_dirsum_test::test_se_1(bool s1,
        bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_test::test_se_1(" << s1 << ", " << s2 << ")";

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
    dimensions<4> dimsc(index_range<4>(i4a, i4b));

    block_index_space<2> bisa(dimsa);
    block_index_space<4> bisc(dimsc);

    mask<2> ma; ma[0] = true; ma[1] = true;
    bisa.split(ma, 2);
    bisa.split(ma, 3);
    mask<4> mc; mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true;
    bisc.split(mc, 2);
    bisc.split(mc, 3);

    symmetry<2, double> syma(bisa), symb(bisa);
    symmetry<4, double> symc(bisc), symc_ref(bisc);

    permutation<2> p1; p1.permute(0, 1);
    permutation<4> p2a, p2b, p2c;
    p2a.permute(0, 1); p2b.permute(2, 3);
    p2c.permute(0, 1).permute(2, 3);

    scalar_transf<double> tr0, tr1(-1.);

    syma.insert(se_perm<2, double>(p1, s1 ? tr0 : tr1));
    symb.insert(se_perm<2, double>(p1, s2 ? tr0 : tr1));
    if (s1)
        symc_ref.insert(
                se_perm<4, double>(p2a, s1 ? tr0 : tr1));
    if (s2)
        symc_ref.insert(
                se_perm<4, double>(p2b, s2 ? tr0 : tr1));
    if (! s1 && ! s2)
        symc_ref.insert(se_perm<4, double>(p2c, s1 ? tr0 : tr1));

    so_dirsum<2, 2, double>(syma, symb).perform(symc);

    symmetry<4, double>::iterator i = symc.begin();
    if(i == symc.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "i == symc.end()");
    }

    compare_ref<4>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

}


/** \test Direct product of a partition symmetry in 2-space and a partition
        symmetry in 2-space to from a 2-space.
 **/
void so_dirsum_test::test_se_2(bool s1,
        bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_test::test_se_2(" << s1 << ", " << s2 << ")";

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
    dimensions<4> dimsc(index_range<4>(i4a, i4b));

    block_index_space<2> bisa(dimsa);
    block_index_space<4> bisc(dimsc);

    mask<2> ma, ma1, ma2;
    ma[0] = ma[1] = true; ma1[0] = ma2[1] = true;
    bisa.split(ma1, 2);
    bisa.split(ma1, 3);
    bisa.split(ma1, 5);
    bisa.split(ma2, 3);
    mask<4> mc, mc1, mc2;
    mc[0] = mc[1] = mc[2] = mc[3] = true;
    mc1[0] = mc1[1] = mc2[2] = mc2[3] = true;
    bisc.split(mc1, 2);
    bisc.split(mc1, 3);
    bisc.split(mc1, 5);
    bisc.split(mc2, 3);

    symmetry<2, double> syma(bisa), symb(bisa);
    symmetry<4, double> symc(bisc), symc_ref(bisc);

    scalar_transf<double> tra(s1 ? 1.0 : -1.0), trb(s2 ? 1.0 : -1.0);

    index<2> i00, i01, i10, i11;
    i01[1] = 1; i10[0] = 1;
    i11[1] = 1; i11[0] = 1;

    se_part<2, double> spa(bisa, ma, 2), spb(bisa, ma, 2);
    spa.add_map(i00, i11, tra);
    spa.add_map(i01, i10, tra);
    spb.add_map(i00, i11, trb);
    spb.add_map(i01, i10, trb);

    index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
    i1110[0] = i1110[1] = i1110[2] = i0001[3] = 1;
    i1101[0] = i1101[1] = i0010[2] = i1101[3] = 1;
    i1100[0] = i1100[1] = i0011[2] = i0011[3] = 1;
    i1011[0] = i0100[1] = i1011[2] = i1011[3] = 1;
    i1010[0] = i0101[1] = i1010[2] = i0101[3] = 1;
    i1001[0] = i0110[1] = i0110[2] = i1001[3] = 1;
    i1000[0] = i0111[1] = i0111[2] = i0111[3] = 1;
    i1111[0] = i1111[1] = i1111[2] = i1111[3] = 1;

    se_part<4, double> spc(bisc, mc, 2);
    if (s2) {
        spc.add_map(i0000, i0101, trb);
        spc.add_map(i0010, i0111, trb);
        spc.add_map(i1000, i1101, trb);
        spc.add_map(i1010, i1111, trb);
        spc.add_map(i0001, i0100, trb);
        spc.add_map(i0011, i0110, trb);
        spc.add_map(i1001, i1100, trb);
        spc.add_map(i1011, i1110, trb);
    }
    if (s1) {
        spc.add_map(i0000, i1010, tra);
        spc.add_map(i0001, i1011, tra);
        spc.add_map(i0100, i1110, tra);
        spc.add_map(i0101, i1111, tra);
        spc.add_map(i0010, i1000, tra);
        spc.add_map(i0011, i1001, tra);
        spc.add_map(i0110, i1100, tra);
        spc.add_map(i0111, i1101, tra);
    }
    if (! s1 && s1 == s2) {
        spc.add_map(i0000, i1111, tra);
        spc.add_map(i0001, i1110, tra);
        spc.add_map(i0010, i1101, tra);
        spc.add_map(i0011, i1100, tra);
        spc.add_map(i0100, i1011, tra);
        spc.add_map(i0101, i1010, tra);
        spc.add_map(i0110, i1001, tra);
        spc.add_map(i1000, i0111, tra);
        spc.add_map(i0111, i1000, tra);
    }
    syma.insert(spa);
    symb.insert(spb);
    symc_ref.insert(spc);

    permutation<4> px; px.permute(1, 2);
    so_dirsum<2, 2, double>(syma, symb, px).perform(symc);

    symmetry<4, double>::iterator i = symc.begin();
    if(i == symc.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "i == symc.end()");
    }

    compare_ref<4>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

}


/** \test Direct product of a label symmetry in 3-space and a label symmetry
        in 3-space to form a 5-space
 **/
void so_dirsum_test::test_se_3() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_se_3()";

    try {

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry with all symmetry elements in 2-space
        and a symmetry with all symmetry elements in 3-space to form a 5-space.
 **/
void so_dirsum_test::test_se_4() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_se_4()";

    try {

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry in 2-space and a symmetry in 3-space
        form a 5-space with a permutation [01234->03214].
 **/
void so_dirsum_test::test_perm_1() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_perm_1()";

    try {

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry in 3-space and a symmetry in 2-space
        form a 5-space with a permutation [01234->21304].
 **/
void so_dirsum_test::test_perm_2() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_perm_2()";

    try {

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry in 3-space and a symmetry in 0-space
        to form a 3-space.
 **/
void so_dirsum_test::test_vac_1() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_vac_1()";

    try {

        index<3> i3a, i3b; i3b[0] = 5; i3b[1] = 5; i3b[2] = 5;
        dimensions<3> dimsa(index_range<3>(i3a, i3b));
        index<0> i0a, i0b;
        dimensions<0> dimsb(index_range<0>(i0a, i0b));

        block_index_space<3> bisa(dimsa);
        block_index_space<0> bisb(dimsb);

        mask<3> ma; ma[0] = true; ma[1] = true; ma[2] = true;
        bisa.split(ma, 2); bisa.split(ma, 3);

        symmetry<3, double> syma(bisa);
        symmetry<0, double> symb(bisb);
        symmetry<3, double> symc(bisa);

        so_dirsum<3, 0, double>(syma, symb).perform(symc);

        symmetry<3, double>::iterator i = symc.begin();
        if(i != symc.end()) {
            fail_test(testname, __FILE__, __LINE__, "i != symc.end()");
        }

        compare_ref<3>::compare(testname, symc, syma);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Direct product of a symmetry in 0-space and a symmetry in 3-space
        to form a 3-space.
 **/
void so_dirsum_test::test_vac_2() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_test::test_vac_2()";

    try {

        index<0> i0a, i0b;
        dimensions<0> dimsa(index_range<0>(i0a, i0b));
        index<3> i3a, i3b; i3b[0] = 5; i3b[1] = 5; i3b[2] = 5;
        dimensions<3> dimsb(index_range<3>(i3a, i3b));

        block_index_space<0> bisa(dimsa);
        block_index_space<3> bisb(dimsb);

        mask<3> mb; mb[0] = true; mb[1] = true; mb[2] = true;
        bisb.split(mb, 2); bisb.split(mb, 3);

        symmetry<0, double> syma(bisa);
        symmetry<3, double> symb(bisb);
        symmetry<3, double> symc(bisb);

        so_dirsum<0, 3, double>(syma, symb).perform(symc);

        symmetry<3, double>::iterator i = symc.begin();
        if(i != symc.end()) {
            fail_test(testname, __FILE__, __LINE__, "i != symc.end()");
        }

        compare_ref<3>::compare(testname, symc, symb);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


} // namespace libtensor
