#include <libtensor/core/tensor_transf_double.h>
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


} // namespace libtensor

