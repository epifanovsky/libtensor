#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_dirprod.h>
#include "../compare_ref.h"
#include "so_dirprod_test.h"

namespace libtensor {


void so_dirprod_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2(true); test_empty_2(false);
    test_empty_3(true); test_empty_3(false);
    test_se_1( true, true); test_se_1( true, false);
    test_se_1(false, true); test_se_1(false, false);
    test_se_2( true, true); test_se_2( true, false);
    test_se_2(false, true); test_se_2(false, false);
    test_se_3();
    test_se_4();
    test_perm_1( true, true); test_perm_1( true, false);
    test_perm_1(false, true); test_perm_1(false, false);
    test_perm_2( true, true); test_perm_2( true, false);
    test_perm_2(false, true); test_perm_2(false, false);
    test_vac_1();
    test_vac_2();
}


/** \test Direct product of empty symmetry in 2-space and empty symmetry in
        1-space to form a 3-space. Expects empty symmetry in 3-space.
 **/
void so_dirprod_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname = "so_dirprod_test::test_empty_1()";

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
        so_dirprod<2, 1, double>(syma, symb).perform(symc);

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
void so_dirprod_test::test_empty_2(bool s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_test::test_empty_2(" << s << ")";

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
        symmetry<3, double> symc(bisc), symc_ref(bisc);

        scalar_transf<double> tr0, tr1(-1.);
        permutation<2> p1; p1.permute(0, 1);
        permutation<3> p2; p2.permute(0, 1);
        syma.insert(se_perm<2, double>(p1, s ? tr0 : tr1));
        symc_ref.insert(se_perm<3, double>(p2, s ? tr0 : tr1));

        so_dirprod<2, 1, double>(syma, symb).perform(symc);

        compare_ref<3>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Direct product of empty symmetry in 1-space and non-empty symmetry in
        2-space to form a 3-space. Expects non-empty symmetry in 3-space.
 **/
void so_dirprod_test::test_empty_3(bool s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_test::test_empty_3(" << s << ")";

    try {

        index<1> i1a, i1b; i1b[0] = 10;
        dimensions<1> dimsa(index_range<1>(i1a, i1b));
        index<2> i2a, i2b; i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dimsb(index_range<2>(i2a, i2b));
        index<3> i3a, i3b; i3b[0] = 10; i3b[1] = 5; i3b[2] = 5;
        dimensions<3> dimsc(index_range<3>(i3a, i3b));

        block_index_space<1> bisa(dimsa);
        block_index_space<2> bisb(dimsb);
        block_index_space<3> bisc(dimsc);

        mask<1> ma; ma[0] = true;
        bisa.split(ma, 5);
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc1, mc2; mc1[1] = true; mc1[2] = true; mc2[0] = true;
        bisc.split(mc1, 2); bisc.split(mc1, 3); bisc.split(mc2, 5);

        symmetry<1, double> syma(bisa);
        symmetry<2, double> symb(bisb);
        symmetry<3, double> symc(bisc), symc_ref(bisc);

        scalar_transf<double> tr0, tr1(-1.);
        permutation<2> p1; p1.permute(0, 1);
        permutation<3> p2; p2.permute(1, 2);
        symb.insert(se_perm<2, double>(p1, s ? tr0 : tr1));
        symc_ref.insert(se_perm<3, double>(p2, s ? tr0 : tr1));

        so_dirprod<1, 2, double>(syma, symb).perform(symc);

        compare_ref<3>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of two S2 symmetries in 2-space forming a 4-space.
        Expects S2 * S2 in 4-space.
 **/
void so_dirprod_test::test_se_1(
        bool s1, bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_test::test_se_1(" << s1 << ", " << s2 << ")";

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

        scalar_transf<double> tr1(s1 ? 1.0 : -1.0), tr2(s2 ? 1.0 : -1.0);
        se_perm<2, double> e1a(permutation<2>().permute(0, 1), tr1);
        se_perm<2, double> e1b(permutation<2>().permute(0, 1), tr2);
        se_perm<4, double> e2a(permutation<4>().permute(0, 1), tr1);
        se_perm<4, double> e2b(permutation<4>().permute(2, 3), tr2);
        syma.insert(e1a);
        symb.insert(e1b);
        symc_ref.insert(e2a);
        symc_ref.insert(e2b);

        so_dirprod<2, 2, double>(syma, symb).perform(symc);

        symmetry<4, double>::iterator i = symc.begin();
        if(i == symc.end()) {
            fail_test(tnss.str().c_str(),
                    __FILE__, __LINE__, "i == symc.end()");
        }

        compare_ref<4>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct product of a partition symmetry in 2-space and a partition
        symmetry in 3-space to from a 5-space.
 **/
void so_dirprod_test::test_se_2(
        bool s1, bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_test::test_se_2(" << s1 << ", " << s2 << ")";

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dimsa(index_range<2>(i2a, i2b));
        index<3> i3a, i3b;
        i3b[0] = 5; i3b[1] = 5; i3b[2] = 5;
        dimensions<3> dimsb(index_range<3>(i3a, i3b));
        index<5> i5a, i5b;
        i5b[0] = 5; i5b[1] = 5; i5b[2] = 5; i5b[3] = 5; i5b[4] = 5;
        dimensions<5> dimsc(index_range<5>(i5a, i5b));

        block_index_space<2> bisa(dimsa);
        block_index_space<3> bisb(dimsb);
        block_index_space<5> bisc(dimsc);

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);
        mask<3> mb; mb[0] = true; mb[1] = true; mb[2] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);
        mask<5> mc;
        mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true; mc[4] = true;
        bisc.split(mc, 2);
        bisc.split(mc, 3);
        bisc.split(mc, 5);

        symmetry<2, double> syma(bisa);
        symmetry<3, double> symb(bisb);
        symmetry<5, double> symc(bisc), symc_ref(bisc);

        se_part<2, double> ela(bisa, ma, 2);
        index<2> i00a, i01a, i02a, i03a;
        scalar_transf<double> tr0, tr1(-1.);
        i02a[0] = 1; i01a[1] = 1;
        i03a[0] = 1; i03a[1] = 1;
        ela.add_map(i00a, i03a, s1 ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);

        se_part<3, double> elb(bisb, mb, 2);
        index<3> i00b, i01b, i02b, i03b, i04b, i05b, i06b, i07b;
        i04b[0] = 1; i03b[1] = 1; i03b[2] = 1;
        i05b[0] = 1; i02b[1] = 1; i05b[2] = 1;
        i06b[0] = 1; i06b[1] = 1; i01b[2] = 1;
        i07b[0] = 1; i07b[1] = 1; i07b[2] = 1;
        elb.add_map(i00b, i07b, s2 ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);
        elb.mark_forbidden(i03b); elb.mark_forbidden(i04b);
        elb.mark_forbidden(i05b); elb.mark_forbidden(i06b);

        se_part<5, double> elc(bisc, mc, 2);
        index<5> i00c, i01c, i02c, i03c, i04c, i05c, i06c, i07c,
            i08c, i09c, i10c, i11c, i12c, i13c, i14c, i15c,
            i16c, i17c, i18c, i19c, i20c, i21c, i22c, i23c,
            i24c, i25c, i26c, i27c, i28c, i29c, i30c, i31c;
        i16c[0] = 1; i15c[1] = 1; i15c[2] = 1; i15c[3] = 1; i15c[4] = 1;
        i17c[0] = 1; i14c[1] = 1; i14c[2] = 1; i14c[3] = 1; i17c[4] = 1;
        i18c[0] = 1; i13c[1] = 1; i13c[2] = 1; i18c[3] = 1; i13c[4] = 1;
        i19c[0] = 1; i12c[1] = 1; i12c[2] = 1; i19c[3] = 1; i19c[4] = 1;
        i20c[0] = 1; i11c[1] = 1; i20c[2] = 1; i11c[3] = 1; i11c[4] = 1;
        i21c[0] = 1; i10c[1] = 1; i21c[2] = 1; i10c[3] = 1; i21c[4] = 1;
        i22c[0] = 1; i09c[1] = 1; i22c[2] = 1; i22c[3] = 1; i09c[4] = 1;
        i23c[0] = 1; i08c[1] = 1; i23c[2] = 1; i23c[3] = 1; i23c[4] = 1;
        i24c[0] = 1; i24c[1] = 1; i07c[2] = 1; i07c[3] = 1; i07c[4] = 1;
        i25c[0] = 1; i25c[1] = 1; i06c[2] = 1; i06c[3] = 1; i25c[4] = 1;
        i26c[0] = 1; i26c[1] = 1; i05c[2] = 1; i26c[3] = 1; i05c[4] = 1;
        i27c[0] = 1; i27c[1] = 1; i04c[2] = 1; i27c[3] = 1; i27c[4] = 1;
        i28c[0] = 1; i28c[1] = 1; i28c[2] = 1; i03c[3] = 1; i03c[4] = 1;
        i29c[0] = 1; i29c[1] = 1; i29c[2] = 1; i02c[3] = 1; i29c[4] = 1;
        i30c[0] = 1; i30c[1] = 1; i30c[2] = 1; i30c[3] = 1; i01c[4] = 1;
        i31c[0] = 1; i31c[1] = 1; i31c[2] = 1; i31c[3] = 1; i31c[4] = 1;
        elc.add_map(i00c, i07c, s2 ? tr0 : tr1);
        elc.add_map(i07c, i24c, s1 == s2 ? tr0 : tr1);
        elc.add_map(i24c, i31c, s2 ? tr0 : tr1);
        elc.mark_forbidden(i01c); elc.mark_forbidden(i02c);
        elc.mark_forbidden(i03c); elc.mark_forbidden(i04c);
        elc.mark_forbidden(i05c); elc.mark_forbidden(i06c);
        elc.mark_forbidden(i08c); elc.mark_forbidden(i09c);
        elc.mark_forbidden(i10c); elc.mark_forbidden(i11c);
        elc.mark_forbidden(i12c); elc.mark_forbidden(i13c);
        elc.mark_forbidden(i14c); elc.mark_forbidden(i15c);
        elc.mark_forbidden(i16c); elc.mark_forbidden(i17c);
        elc.mark_forbidden(i18c); elc.mark_forbidden(i19c);
        elc.mark_forbidden(i20c); elc.mark_forbidden(i21c);
        elc.mark_forbidden(i22c); elc.mark_forbidden(i23c);
        elc.mark_forbidden(i25c); elc.mark_forbidden(i26c);
        elc.mark_forbidden(i27c); elc.mark_forbidden(i28c);
        elc.mark_forbidden(i29c); elc.mark_forbidden(i30c);

        syma.insert(ela);
        symb.insert(elb);
        symc_ref.insert(elc);

        so_dirprod<2, 3, double>(syma, symb).perform(symc);

        symmetry<5, double>::iterator i = symc.begin();
        if(i == symc.end()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__, "i == symc.end()");
        }

        compare_ref<5>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

}


/** \test Direct product of a label symmetry in 2-space and a label symmetry
        in 3-space to form a 5-space
 **/
void so_dirprod_test::test_se_3() throw(libtest::test_exception) {

    static const char *testname = "so_dirprod_test::test_se_3()";

    try {

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry with all symmetry elements in 2-space
        and a symmetry with all symmetry elements in 3-space to form a 5-space.
 **/
void so_dirprod_test::test_se_4() throw(libtest::test_exception) {

    static const char *testname = "so_dirprod_test::test_se_4()";

    try {

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry in 2-space and a symmetry in 3-space
        form a 5-space with a permutation [01234->03214].
 **/
void so_dirprod_test::test_perm_1(
        bool s1, bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_test::test_perm_1(" << s1 << ", " << s2 << ")";

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dimsa(index_range<2>(i2a, i2b));
        index<3> i3a, i3b;
        i3b[0] = 5; i3b[1] = 5; i3b[2] = 5;
        dimensions<3> dimsb(index_range<3>(i3a, i3b));
        index<5> i5a, i5b;
        i5b[0] = 5; i5b[1] = 5; i5b[2] = 5; i5b[3] = 5; i5b[4] = 5;
        dimensions<5> dimsc(index_range<5>(i5a, i5b));

        block_index_space<2> bisa(dimsa);
        block_index_space<3> bisb(dimsb);
        block_index_space<5> bisc(dimsc);

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);
        mask<3> mb; mb[0] = true; mb[1] = true; mb[2] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);
        mask<5> mc;
        mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true; mc[4] = true;
        bisc.split(mc, 2);
        bisc.split(mc, 3);
        bisc.split(mc, 5);

        symmetry<2, double> syma(bisa);
        symmetry<3, double> symb(bisb);
        symmetry<5, double> symc(bisc), symc_ref(bisc);

        se_part<2, double> ela(bisa, ma, 2);
        index<2> i00a, i01a, i02a, i03a;
        scalar_transf<double> tr0, tr1(-1.);
        i02a[0] = 1; i01a[1] = 1;
        i03a[0] = 1; i03a[1] = 1;
        ela.add_map(i00a, i03a, s1 ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);

        se_part<3, double> elb(bisb, mb, 2);
        index<3> i00b, i01b, i02b, i03b, i04b, i05b, i06b, i07b;
        i04b[0] = 1; i03b[1] = 1; i03b[2] = 1;
        i05b[0] = 1; i02b[1] = 1; i05b[2] = 1;
        i06b[0] = 1; i06b[1] = 1; i01b[2] = 1;
        i07b[0] = 1; i07b[1] = 1; i07b[2] = 1;
        elb.add_map(i00b, i07b, s2 ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);
        elb.mark_forbidden(i03b); elb.mark_forbidden(i04b);
        elb.mark_forbidden(i05b); elb.mark_forbidden(i06b);

        se_part<5, double> elc(bisc, mc, 2);
        index<5> i00c, i01c, i02c, i03c, i04c, i05c, i06c, i07c,
            i08c, i09c, i10c, i11c, i12c, i13c, i14c, i15c,
            i16c, i17c, i18c, i19c, i20c, i21c, i22c, i23c,
            i24c, i25c, i26c, i27c, i28c, i29c, i30c, i31c;
        i16c[0] = 1; i15c[1] = 1; i15c[2] = 1; i15c[3] = 1; i15c[4] = 1;
        i17c[0] = 1; i14c[1] = 1; i14c[2] = 1; i14c[3] = 1; i17c[4] = 1;
        i18c[0] = 1; i13c[1] = 1; i13c[2] = 1; i18c[3] = 1; i13c[4] = 1;
        i19c[0] = 1; i12c[1] = 1; i12c[2] = 1; i19c[3] = 1; i19c[4] = 1;
        i20c[0] = 1; i11c[1] = 1; i20c[2] = 1; i11c[3] = 1; i11c[4] = 1;
        i21c[0] = 1; i10c[1] = 1; i21c[2] = 1; i10c[3] = 1; i21c[4] = 1;
        i22c[0] = 1; i09c[1] = 1; i22c[2] = 1; i22c[3] = 1; i09c[4] = 1;
        i23c[0] = 1; i08c[1] = 1; i23c[2] = 1; i23c[3] = 1; i23c[4] = 1;
        i24c[0] = 1; i24c[1] = 1; i07c[2] = 1; i07c[3] = 1; i07c[4] = 1;
        i25c[0] = 1; i25c[1] = 1; i06c[2] = 1; i06c[3] = 1; i25c[4] = 1;
        i26c[0] = 1; i26c[1] = 1; i05c[2] = 1; i26c[3] = 1; i05c[4] = 1;
        i27c[0] = 1; i27c[1] = 1; i04c[2] = 1; i27c[3] = 1; i27c[4] = 1;
        i28c[0] = 1; i28c[1] = 1; i28c[2] = 1; i03c[3] = 1; i03c[4] = 1;
        i29c[0] = 1; i29c[1] = 1; i29c[2] = 1; i02c[3] = 1; i29c[4] = 1;
        i30c[0] = 1; i30c[1] = 1; i30c[2] = 1; i30c[3] = 1; i01c[4] = 1;
        i31c[0] = 1; i31c[1] = 1; i31c[2] = 1; i31c[3] = 1; i31c[4] = 1;
        elc.add_map(i00c, i07c, s2 ? tr0 : tr1);
        elc.add_map(i07c, i24c, s1 == s2 ? tr0 : tr1);
        elc.add_map(i24c, i31c, s2 ? tr0 : tr1);
        elc.mark_forbidden(i01c); elc.mark_forbidden(i02c);
        elc.mark_forbidden(i03c); elc.mark_forbidden(i04c);
        elc.mark_forbidden(i05c); elc.mark_forbidden(i06c);
        elc.mark_forbidden(i08c); elc.mark_forbidden(i09c);
        elc.mark_forbidden(i10c); elc.mark_forbidden(i11c);
        elc.mark_forbidden(i12c); elc.mark_forbidden(i13c);
        elc.mark_forbidden(i14c); elc.mark_forbidden(i15c);
        elc.mark_forbidden(i16c); elc.mark_forbidden(i17c);
        elc.mark_forbidden(i18c); elc.mark_forbidden(i19c);
        elc.mark_forbidden(i20c); elc.mark_forbidden(i21c);
        elc.mark_forbidden(i22c); elc.mark_forbidden(i23c);
        elc.mark_forbidden(i25c); elc.mark_forbidden(i26c);
        elc.mark_forbidden(i27c); elc.mark_forbidden(i28c);
        elc.mark_forbidden(i29c); elc.mark_forbidden(i30c);

        syma.insert(ela);
        symb.insert(elb);
        symc_ref.insert(elc);

        permutation<5> pc;
        pc.permute(1, 3);
        so_dirprod<2, 3, double>(syma, symb, pc).perform(symc);

        symmetry<5, double>::iterator i = symc.begin();
        if(i == symc.end()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__, "i == symc.end()");
        }

        compare_ref<5>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry in 3-space and a symmetry in 2-space
        form a 5-space with a permutation [01234->21304].
 **/
void so_dirprod_test::test_perm_2(
        bool s1, bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_test::test_perm_2(" << s1 << ", " << s2 << ")";

    try {

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        dimensions<2> dimsa(index_range<2>(i2a, i2b));
        index<3> i3a, i3b;
        i3b[0] = 5; i3b[1] = 5; i3b[2] = 5;
        dimensions<3> dimsb(index_range<3>(i3a, i3b));
        index<5> i5a, i5b;
        i5b[0] = 5; i5b[1] = 5; i5b[2] = 5; i5b[3] = 5; i5b[4] = 5;
        dimensions<5> dimsc(index_range<5>(i5a, i5b));

        block_index_space<2> bisa(dimsa);
        block_index_space<3> bisb(dimsb);
        block_index_space<5> bisc(dimsc);

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);
        mask<3> mb; mb[0] = true; mb[1] = true; mb[2] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);
        mask<5> mc;
        mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true; mc[4] = true;
        bisc.split(mc, 2);
        bisc.split(mc, 3);
        bisc.split(mc, 5);

        symmetry<2, double> syma(bisa);
        symmetry<3, double> symb(bisb);
        symmetry<5, double> symc(bisc), symc_ref(bisc);

        se_part<2, double> ela(bisa, ma, 2);
        scalar_transf<double> tr0, tr1(-1.);
        index<2> i00a, i01a, i02a, i03a;
        i02a[0] = 1; i01a[1] = 1;
        i03a[0] = 1; i03a[1] = 1;
        ela.add_map(i00a, i03a, s1 ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);

        se_part<3, double> elb(bisb, mb, 2);
        index<3> i00b, i01b, i02b, i03b, i04b, i05b, i06b, i07b;
        i04b[0] = 1; i03b[1] = 1; i03b[2] = 1;
        i05b[0] = 1; i02b[1] = 1; i05b[2] = 1;
        i06b[0] = 1; i06b[1] = 1; i01b[2] = 1;
        i07b[0] = 1; i07b[1] = 1; i07b[2] = 1;
        elb.add_map(i00b, i07b, s2 ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);
        elb.mark_forbidden(i03b); elb.mark_forbidden(i04b);
        elb.mark_forbidden(i05b); elb.mark_forbidden(i06b);

        se_part<5, double> elc(bisc, mc, 2);
        index<5> i00c, i01c, i02c, i03c, i04c, i05c, i06c, i07c,
            i08c, i09c, i10c, i11c, i12c, i13c, i14c, i15c,
            i16c, i17c, i18c, i19c, i20c, i21c, i22c, i23c,
            i24c, i25c, i26c, i27c, i28c, i29c, i30c, i31c;
        i16c[0] = 1; i15c[1] = 1; i15c[2] = 1; i15c[3] = 1; i15c[4] = 1;
        i17c[0] = 1; i14c[1] = 1; i14c[2] = 1; i14c[3] = 1; i17c[4] = 1;
        i18c[0] = 1; i13c[1] = 1; i13c[2] = 1; i18c[3] = 1; i13c[4] = 1;
        i19c[0] = 1; i12c[1] = 1; i12c[2] = 1; i19c[3] = 1; i19c[4] = 1;
        i20c[0] = 1; i11c[1] = 1; i20c[2] = 1; i11c[3] = 1; i11c[4] = 1;
        i21c[0] = 1; i10c[1] = 1; i21c[2] = 1; i10c[3] = 1; i21c[4] = 1;
        i22c[0] = 1; i09c[1] = 1; i22c[2] = 1; i22c[3] = 1; i09c[4] = 1;
        i23c[0] = 1; i08c[1] = 1; i23c[2] = 1; i23c[3] = 1; i23c[4] = 1;
        i24c[0] = 1; i24c[1] = 1; i07c[2] = 1; i07c[3] = 1; i07c[4] = 1;
        i25c[0] = 1; i25c[1] = 1; i06c[2] = 1; i06c[3] = 1; i25c[4] = 1;
        i26c[0] = 1; i26c[1] = 1; i05c[2] = 1; i26c[3] = 1; i05c[4] = 1;
        i27c[0] = 1; i27c[1] = 1; i04c[2] = 1; i27c[3] = 1; i27c[4] = 1;
        i28c[0] = 1; i28c[1] = 1; i28c[2] = 1; i03c[3] = 1; i03c[4] = 1;
        i29c[0] = 1; i29c[1] = 1; i29c[2] = 1; i02c[3] = 1; i29c[4] = 1;
        i30c[0] = 1; i30c[1] = 1; i30c[2] = 1; i30c[3] = 1; i01c[4] = 1;
        i31c[0] = 1; i31c[1] = 1; i31c[2] = 1; i31c[3] = 1; i31c[4] = 1;
        elc.add_map(i00c, i07c, s2 ? tr0 : tr1);
        elc.add_map(i07c, i24c, s1 == s2 ? tr0 : tr1);
        elc.add_map(i24c, i31c, s2 ? tr0 : tr1);
        elc.mark_forbidden(i01c); elc.mark_forbidden(i02c);
        elc.mark_forbidden(i03c); elc.mark_forbidden(i04c);
        elc.mark_forbidden(i05c); elc.mark_forbidden(i06c);
        elc.mark_forbidden(i08c); elc.mark_forbidden(i09c);
        elc.mark_forbidden(i10c); elc.mark_forbidden(i11c);
        elc.mark_forbidden(i12c); elc.mark_forbidden(i13c);
        elc.mark_forbidden(i14c); elc.mark_forbidden(i15c);
        elc.mark_forbidden(i16c); elc.mark_forbidden(i17c);
        elc.mark_forbidden(i18c); elc.mark_forbidden(i19c);
        elc.mark_forbidden(i20c); elc.mark_forbidden(i21c);
        elc.mark_forbidden(i22c); elc.mark_forbidden(i23c);
        elc.mark_forbidden(i25c); elc.mark_forbidden(i26c);
        elc.mark_forbidden(i27c); elc.mark_forbidden(i28c);
        elc.mark_forbidden(i29c); elc.mark_forbidden(i30c);

        syma.insert(ela);
        symb.insert(elb);
        symc_ref.insert(elc);

        permutation<5> pc;
        pc.permute(0, 2).permute(2, 3);
        so_dirprod<2, 3, double>(syma, symb, pc).perform(symc);

        symmetry<5, double>::iterator i = symc.begin();
        if(i == symc.end()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__, "i == symc.end()");
        }

        compare_ref<5>::compare(tnss.str().c_str(), symc, symc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

}

/** \test Direct product of a symmetry in 3-space and a symmetry in 0-space
        to form a 3-space.
 **/
void so_dirprod_test::test_vac_1() throw(libtest::test_exception) {

    static const char *testname = "so_dirprod_test::test_vac_1()";

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

        so_dirprod<3, 0, double>(syma, symb).perform(symc);

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
void so_dirprod_test::test_vac_2() throw(libtest::test_exception) {

    static const char *testname = "so_dirprod_test::test_vac_2()";

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

        so_dirprod<0, 3, double>(syma, symb).perform(symc);

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
