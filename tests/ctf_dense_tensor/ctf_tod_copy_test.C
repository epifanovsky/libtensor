#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_copy.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_copy_test.h"

namespace libtensor {


void ctf_tod_copy_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();
        test_2a();
        test_2b();
        test_3a();
        test_3b();
        test_4a();
        test_4b();
        test_5();
        test_6();
        test_7();
        test_8();
        test_9();
        test_10();
        test_11();
        test_12();
        test_13();
        test_14();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_copy_test::test_1a() {

    static const char testname[] = "ctf_tod_copy_test::test_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_1b() {

    static const char testname[] = "ctf_tod_copy_test::test_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(tb).perform(true, tb_ref);
    tod_copy<2>(ta, -0.5).perform(false, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, -0.5).perform(false, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_2a() {

    static const char testname[] = "ctf_tod_copy_test::test_2a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta, perma).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, perma).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_2b() {

    static const char testname[] = "ctf_tod_copy_test::test_2b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(tb).perform(true, tb_ref);
    tod_copy<2>(ta, perma, -0.5).perform(false, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, perma, -0.5).perform(false, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_3a() {

    static const char testname[] = "ctf_tod_copy_test::test_3a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tensor_transf<2, double> tra(perma, scalar_transf<double>(2.0));

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta, tra).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, tra).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_3b() {

    static const char testname[] = "ctf_tod_copy_test::test_3b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tensor_transf<2, double> tra(perma, scalar_transf<double>(2.0));

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(tb).perform(true, tb_ref);
    tod_copy<2>(ta, tra).perform(false, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, tra).perform(false, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_4a() {

    static const char testname[] = "ctf_tod_copy_test::test_4a()";

    typedef allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    permutation<3> perma;
    perma.permute(0, 1).permute(1, 2);
    dimsb.permute(perma);
    dense_tensor<3, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<3, double> dta(dimsa), dtb(dimsb);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);
    tod_copy<3>(ta, perma, 0.5).perform(true, tb_ref);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_copy<3>(dta, perma, 0.5).perform(true, dtb);
    ctf_tod_collect<3>(dtb).perform(tb);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_4b() {

    static const char testname[] = "ctf_tod_copy_test::test_4b()";

    typedef allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    permutation<3> perma;
    perma.permute(0, 2).permute(1, 2);
    dimsb.permute(perma);
    dense_tensor<3, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<3, double> dta(dimsa), dtb(dimsb);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);
    tod_copy<3>(tb).perform(true, tb_ref);
    tod_copy<3>(ta, perma, -1.0).perform(false, tb_ref);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_copy<3>(dta, perma, -1.0).perform(false, dtb);
    ctf_tod_collect<3>(dtb).perform(tb);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_5() {

    static const char testname[] = "ctf_tod_copy_test::test_5()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 19; i2[1] = 19;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<2, unsigned> symb_grp(0), symb_sym(0);
    ctf_symmetry<2, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<2, double> dta(dims), dtb(dims, symb);

    tod_random<2>().perform(tt);
    tod_random<2>().perform(tb);
    tod_copy<2>(tt).perform(true, ta);
    tod_copy<2>(tt, permutation<2>().permute(0, 1)).perform(false, ta);
    tod_copy<2>(ta).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);

    ctf_tod_copy<2>(dta).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_6() {

    static const char testname[] = "ctf_tod_copy_test::test_6()";

    typedef allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 19;
    dimensions<3> dims(index_range<3>(i1, i2));
    dense_tensor<3, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<3, unsigned> symb_grp(0), symb_sym(0);
    ctf_symmetry<3, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<3, double> dta(dims), dtb(dims, symb);

    tod_random<3>().perform(tt);
    tod_random<3>().perform(tb);
    tod_copy<3>(tt).perform(true, ta);
    tod_copy<3>(tt, permutation<3>().permute(0, 1)).perform(false, ta);
    tod_copy<3>(tt, permutation<3>().permute(0, 2)).perform(false, ta);
    tod_copy<3>(tt, permutation<3>().permute(1, 2)).perform(false, ta);
    tod_copy<3>(tt, permutation<3>().permute(0, 1).permute(1, 2)).
        perform(false, ta);
    tod_copy<3>(tt, permutation<3>().permute(1, 2).permute(0, 1)).
        perform(false, ta);
    tod_copy<3>(ta).perform(true, tb_ref);

    ctf_tod_distribute<3>(ta).perform(dta);

    ctf_tod_copy<3>(dta).perform(true, dtb);
    ctf_tod_collect<3>(dtb).perform(tb);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_7() {

    static const char testname[] = "ctf_tod_copy_test::test_7()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<4, unsigned> symb_grp(0), symb_sym(0);
    symb_grp[0] = 0; symb_grp[1] = 0; symb_grp[2] = 1; symb_grp[3] = 1;
    ctf_symmetry<4, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<4, double> dta(dims), dtb(dims, symb);

    tod_random<4>().perform(tt);
    tod_random<4>().perform(tb);
    tod_copy<4>(tt).perform(true, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 1)).perform(false, ta);
    tod_copy<4>(tt, permutation<4>().permute(2, 3)).perform(false, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 1).permute(2, 3)).
        perform(false, ta);
    tod_copy<4>(ta).perform(true, tb_ref);

    ctf_tod_distribute<4>(ta).perform(dta);

    ctf_tod_copy<4>(dta).perform(true, dtb);
    ctf_tod_collect<4>(dtb).perform(tb);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_8() {

    static const char testname[] = "ctf_tod_copy_test::test_8()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<4, unsigned> symb_grp(0), symb_sym(0);
    symb_grp[0] = 0; symb_grp[1] = 1; symb_grp[2] = 0; symb_grp[3] = 1;
    ctf_symmetry<4, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<4, double> dta(dims), dtb(dims, symb);

    tod_random<4>().perform(tt);
    tod_random<4>().perform(tb);
    tod_copy<4>(tt, 2.0).perform(true, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 2)).perform(false, ta);
    tod_copy<4>(tt, permutation<4>().permute(1, 3)).perform(false, ta);
    tod_copy<4>(ta).perform(true, tb_ref);

    ctf_tod_distribute<4>(ta).perform(dta);

    ctf_tod_copy<4>(dta).perform(true, dtb);
    ctf_tod_collect<4>(dtb).perform(tb);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_9() {

    static const char testname[] = "ctf_tod_copy_test::test_9()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<4, unsigned> syma_grp(0), syma_sym(0);
    syma_grp[0] = 0; syma_grp[1] = 0; syma_grp[2] = 1; syma_grp[3] = 2;
    sequence<4, unsigned> symb_grp(0), symb_sym(0);
    symb_grp[0] = 0; symb_grp[1] = 1; symb_grp[2] = 2; symb_grp[3] = 2;
    ctf_symmetry<4, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<4, double> dta(dims), dtb(dims, symb);

    tod_random<4>().perform(tt);
    tod_random<4>().perform(tb);
    tod_copy<4>(tt).perform(true, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 1)).perform(false, ta);
    tod_copy<4>(tt, permutation<4>().permute(2, 3)).perform(false, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 1).permute(2, 3)).
        perform(false, ta);
    tod_copy<4>(ta).perform(true, tb_ref);

    ctf_tod_distribute<4>(ta).perform(dta);

    ctf_tod_copy<4>(dta).perform(true, dtb);
    ctf_tod_collect<4>(dtb).perform(tb);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_10() {

    static const char testname[] = "ctf_tod_copy_test::test_10()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<4, unsigned> syma_grp(0), syma_sym(0);
    syma_grp[0] = 0; syma_grp[1] = 0; syma_grp[2] = 1; syma_grp[3] = 2;
    ctf_symmetry<4, double> syma(syma_grp, syma_sym);
    sequence<4, unsigned> symb_grp(0), symb_sym(0);
    symb_grp[0] = 0; symb_grp[1] = 1; symb_grp[2] = 2; symb_grp[3] = 2;
    ctf_symmetry<4, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<4, double> dta(dims, syma), dtb(dims, symb);

    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 1)).perform(false, ta);
    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, tb);
    tod_copy<4>(tt, permutation<4>().permute(2, 3)).perform(false, tb);

    ctf_tod_distribute<4>(ta).perform(dta);
    ctf_tod_distribute<4>(tb).perform(dtb);

    tod_copy<4>(tb).perform(true, tb_ref);
    tod_copy<4>(ta, permutation<4>().permute(0, 2).permute(1, 3), -0.5).
        perform(true, tb_ref);

    ctf_tod_copy<4>(dta, permutation<4>().permute(0, 2).permute(1, 3), -0.5).
        perform(true, dtb);
    ctf_tod_collect<4>(dtb).perform(tb);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_11() {

    static const char testname[] = "ctf_tod_copy_test::test_11()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    sequence<2, unsigned> sym_grp(0), sym_sym(0);
    ctf_symmetry<2, double> sym(sym_grp, sym_sym);
    sym_sym[0] = 1;
    sym.add_component(sym_grp, sym_sym);
    ctf_dense_tensor<2, double> dta(dimsa, sym), dtb(dimsb, sym);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_12() {

    static const char testname[] = "ctf_tod_copy_test::test_12()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    sequence<2, unsigned> sym_grp(0), sym_sym(0);
    ctf_symmetry<2, double> sym(sym_grp, sym_sym);
    sym_sym[0] = 1;
    sym.add_component(sym_grp, sym_sym);
    ctf_dense_tensor<2, double> dta(dimsa, sym), dtb(dimsb, sym);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta, permutation<2>().permute(0, 1)).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, permutation<2>().permute(0, 1)).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_13() {

    static const char testname[] = "ctf_tod_copy_test::test_13()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<4, unsigned> syma_grp(0), syma_sym(0);
    syma_grp[0] = 0; syma_grp[1] = 1; syma_grp[2] = 2; syma_grp[3] = 3;
    ctf_symmetry<4, double> syma(syma_grp, syma_sym);
    sequence<4, unsigned> symb_grp(0), symb_sym(0);
    symb_grp[0] = 0; symb_grp[1] = 0; symb_grp[2] = 1; symb_grp[3] = 1;
    ctf_symmetry<4, double> symb(symb_grp, symb_sym);
    symb_sym[0] = 1; symb_sym[1] = 1;
    symb.add_component(symb_grp, symb_sym);
    ctf_dense_tensor<4, double> dta(dims, syma), dtb(dims, symb);

    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 1).permute(2, 3)).
        perform(false, ta);
    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, tb);
    tod_copy<4>(tt, permutation<4>().permute(0, 1).permute(2, 3)).
        perform(false, tb);

    ctf_tod_distribute<4>(ta).perform(dta);
    ctf_tod_distribute<4>(tb).perform(dtb);

    tod_copy<4>(tb).perform(true, tb_ref);
    tod_copy<4>(ta, -0.5).perform(true, tb_ref);

    ctf_tod_copy<4>(dta, -0.5).perform(true, dtb);
    ctf_tod_collect<4>(dtb).perform(tb);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_14() {

    static const char testname[] = "ctf_tod_copy_test::test_14()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> tt(dims), ta(dims), tb(dims),
        tb_ref(dims);
    sequence<4, unsigned> syma_grp(0), syma_sym(0);
    syma_grp[0] = 0; syma_grp[1] = 1; syma_grp[2] = 0; syma_grp[3] = 3;
    ctf_symmetry<4, double> syma(syma_grp, syma_sym);
    sequence<4, unsigned> symb_grp(0), symb_sym(0);
    symb_grp[0] = 0; symb_grp[1] = 0; symb_grp[2] = 2; symb_grp[3] = 3;
    symb_sym[0] = 1;
    ctf_symmetry<4, double> symb(symb_grp, symb_sym);
    ctf_dense_tensor<4, double> dta(dims, syma), dtb(dims, symb);

    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, ta);
    tod_copy<4>(tt, permutation<4>().permute(0, 2)).perform(false, ta);
    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, tb);
    tod_copy<4>(tt, permutation<4>().permute(0, 1), -1.0).perform(false, tb);

    ctf_tod_distribute<4>(ta).perform(dta);
    ctf_tod_distribute<4>(tb).perform(dtb);

    tod_copy<4>(tb).perform(true, tb_ref);
    tod_copy<4>(ta).perform(false, tb_ref);
    tod_copy<4>(ta, permutation<4>().permute(0, 1), -1.0).
        perform(false, tb_ref);

    ctf_tod_copy<4>(dta).perform(false, dtb);
    ctf_tod_copy<4>(dta, permutation<4>().permute(0, 1), -1.0).
        perform(false, dtb);
    ctf_tod_collect<4>(dtb).perform(tb);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

