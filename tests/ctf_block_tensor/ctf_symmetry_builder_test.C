#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_symmetry_builder.h>
#include "../ctf_dense_tensor/ctf_symmetry_test_equals.h"
#include "ctf_symmetry_builder_test.h"

namespace libtensor {


void ctf_symmetry_builder_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();
        test_2a();
        test_2b();
        test_3();
        test_4();
        test_5();
        test_6();
        test_7();
        test_8();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_symmetry_builder_test::test_1() {

    static const char testname[] = "ctf_symmetry_builder_test::test_1()";

    try {

    index<2> i1, i2, ii;
    i2[0] = 3; i2[1] = 3;
    block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
    symmetry<2, double> sym(bis);

    ii[0] = 0; ii[1] = 0;
    transf_list<2, double> trl(sym, ii);

    sequence<2, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 1; grpind[0] = 0; grpind[1] = 0;
    ctf_symmetry<2, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<2, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_2a() {

    static const char testname[] = "ctf_symmetry_builder_test::test_2a()";

    try {

    index<2> i1, i2, ii;
    i2[0] = 3; i2[1] = 3;
    block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
    symmetry<2, double> sym(bis);
    sym.insert(se_perm<2, double>(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0)));

    ii[0] = 0; ii[1] = 0;
    transf_list<2, double> trl(sym, ii);

    sequence<2, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 0; grpind[0] = 0;
    ctf_symmetry<2, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<2, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_2b() {

    static const char testname[] = "ctf_symmetry_builder_test::test_2b()";

    try {

    index<2> i1, i2, ii;
    i2[0] = 3; i2[1] = 3;
    block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
    symmetry<2, double> sym(bis);
    sym.insert(se_perm<2, double>(permutation<2>().permute(0, 1),
        scalar_transf<double>(-1.0)));

    ii[0] = 0; ii[1] = 0;
    transf_list<2, double> trl(sym, ii);

    sequence<2, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 0; grpind[0] = 1;
    ctf_symmetry<2, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<2, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_3() {

    static const char testname[] = "ctf_symmetry_builder_test::test_3()";

    try {

    index<4> i1, i2, ii;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    symmetry<4, double> sym(bis);
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1),
        scalar_transf<double>(1.0)));
    sym.insert(se_perm<4, double>(
        permutation<4>().permute(0, 1).permute(1, 2).permute(2, 3),
        scalar_transf<double>(1.0)));

    ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
    transf_list<4, double> trl(sym, ii);

    sequence<4, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 0; grp[3] = 0;
    grpind[0] = 0;
    ctf_symmetry<4, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<4, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_4() {

    static const char testname[] = "ctf_symmetry_builder_test::test_4()";

    try {

    index<4> i1, i2, ii;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    symmetry<4, double> sym(bis);
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1),
        scalar_transf<double>(1.0)));
    sym.insert(se_perm<4, double>(permutation<4>().permute(2, 3),
        scalar_transf<double>(-1.0)));

    ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
    transf_list<4, double> trl(sym, ii);

    sequence<4, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 1; grp[3] = 1;
    grpind[0] = 0; grpind[1] = 1;
    ctf_symmetry<4, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<4, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_5() {

    static const char testname[] = "ctf_symmetry_builder_test::test_5()";

    try {

    index<4> i1, i2, ii;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    symmetry<4, double> sym(bis);
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 2),
        scalar_transf<double>(1.0)));
    sym.insert(se_perm<4, double>(permutation<4>().permute(1, 3),
        scalar_transf<double>(-1.0)));

    ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
    transf_list<4, double> trl(sym, ii);

    sequence<4, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 1; grp[2] = 0; grp[3] = 1;
    grpind[0] = 0; grpind[1] = 1;
    ctf_symmetry<4, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<4, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_6() {

    static const char testname[] = "ctf_symmetry_builder_test::test_6()";

    try {

    index<4> i1, i2, ii;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    symmetry<4, double> sym(bis);
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 2).permute(1, 3),
        scalar_transf<double>(1.0)));

    ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
    transf_list<4, double> trl(sym, ii);

    sequence<4, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 1; grp[2] = 2; grp[3] = 3;
    grpind[0] = 0; grpind[1] = 0; grpind[2] = 0; grpind[3] = 0;
    ctf_symmetry<4, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<4, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_7() {

    static const char testname[] = "ctf_symmetry_builder_test::test_7()";

    try {

    index<4> i1, i2, ii;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    symmetry<4, double> sym(bis);
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 2).permute(1, 3),
        scalar_transf<double>(1.0)));
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1),
        scalar_transf<double>(-1.0)));

    ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
    transf_list<4, double> trl(sym, ii);

    sequence<4, unsigned> grp(0), grpind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 1; grp[3] = 1;
    grpind[0] = 1; grpind[1] = 1;
    ctf_symmetry<4, double> dsym_ref(grp, grpind);

    ctf_symmetry_builder<4, double> dsymbld(trl);

    if(!ctf_symmetry_test_equals(dsymbld.get_symmetry(), dsym_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld.get_symmetry() != dsym_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_builder_test::test_8() {

    static const char testname[] = "ctf_symmetry_builder_test::test_8()";

    try {

    index<4> i1, i2, ii;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    bis.split(m1111, 2);
    symmetry<4, double> sym(bis);
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1),
        scalar_transf<double>(-1.0)));
    sym.insert(se_perm<4, double>(permutation<4>().permute(2, 3),
        scalar_transf<double>(-1.0)));
    se_part<4, double> separt(bis, m1111, 2);
    index<4> i0000, i0101, i0110, i1001, i1010, i1111;
    i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
    i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
    i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
    separt.add_map(i0000, i1111);
    separt.add_map(i0101, i1010);
    separt.add_map(i0110, i1001);
    sym.insert(separt);

    ii[0] = 0; ii[1] = 0; ii[2] = 0; ii[3] = 0;
    transf_list<4, double> trl1(sym, ii);

    sequence<4, unsigned> grp1(0), grpind1(0);
    grp1[0] = 0; grp1[1] = 0; grp1[2] = 1; grp1[3] = 1;
    grpind1[0] = 1; grpind1[1] = 1;
    ctf_symmetry<4, double> dsym1_ref(grp1, grpind1);

    ctf_symmetry_builder<4, double> dsymbld1(trl1);

    if(!ctf_symmetry_test_equals(dsymbld1.get_symmetry(), dsym1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld1.get_symmetry() != dsym1_ref");
    }

    ii[0] = 0; ii[1] = 1; ii[2] = 0; ii[3] = 1;
    transf_list<4, double> trl2(sym, ii);

    sequence<4, unsigned> grp2(0), grpind2(0);
    grp2[0] = 0; grp2[1] = 1; grp2[2] = 2; grp2[3] = 3;
    ctf_symmetry<4, double> dsym2_ref(grp2, grpind2, true);

    ctf_symmetry_builder<4, double> dsymbld2(trl2);

    if(!ctf_symmetry_test_equals(dsymbld2.get_symmetry(), dsym2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "dsymbld2.get_symmetry() != dsym2_ref");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

