#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_distribute_test.h"

namespace libtensor {


void ctf_tod_distribute_test::perform() throw(libtest::test_exception) {

    ctf::init();

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();

    ctf::exit();
}


void ctf_tod_distribute_test::test_1() {

    static const char testname[] = "ctf_tod_distribute_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    ctf_dense_tensor<2, double> dt(dims);

    tod_random<2>().perform(t_ref);
    ctf_tod_distribute<2>(t_ref).perform(dt);
    ctf_tod_collect<2>(dt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_distribute_test::test_2() {

    static const char testname[] = "ctf_tod_distribute_test::test_2()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), tt(dims), t_ref(dims);
    sequence<2, unsigned> sym_grp(0), sym_sym(0);
    ctf_symmetry<2, double> sym(sym_grp, sym_sym);
    ctf_dense_tensor<2, double> dt(dims, sym);

    tod_random<2>().perform(tt);
    tod_copy<2>(tt).perform(true, t_ref);
    tod_copy<2>(tt, permutation<2>().permute(0, 1)).perform(false, t_ref);
    ctf_tod_distribute<2>(t_ref).perform(dt);
    ctf_tod_collect<2>(dt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_distribute_test::test_3() {

    static const char testname[] = "ctf_tod_distribute_test::test_3()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    sequence<2, unsigned> sym_grp(0), sym_sym(0);
    ctf_symmetry<2, double> sym(sym_grp, sym_sym);
    sym_sym[0] = 1;
    sym.add_component(sym_grp, sym_sym);
    ctf_dense_tensor<2, double> dt(dims, sym);

    tod_random<2>().perform(t_ref);
    ctf_tod_distribute<2>(t_ref).perform(dt);
    ctf_tod_collect<2>(dt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_distribute_test::test_4() {

    static const char testname[] = "ctf_tod_distribute_test::test_4()";

    typedef allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9;
    dimensions<3> dims(index_range<3>(i1, i2));
    dense_tensor<3, double, allocator_t> t(dims), tt(dims), t_ref(dims);
    sequence<3, unsigned> sym_grp(0), sym_sym(0);
    sym_sym[0] = 1;
    ctf_symmetry<3, double> sym(sym_grp, sym_sym);
    ctf_dense_tensor<3, double> dt(dims, sym);

    permutation<3> p012, p021, p102, p120, p201, p210;
    p021.permute(1, 2);
    p102.permute(0, 1);
    p120.permute(0, 1).permute(1, 2);
    p201.permute(0, 1).permute(0, 2);
    p210.permute(0, 2);
 
    tod_random<3>().perform(tt);
    tod_copy<3>(tt, p012, 1.0).perform(true, t_ref);
    tod_copy<3>(tt, p021, -1.0).perform(false, t_ref);
    tod_copy<3>(tt, p102, -1.0).perform(false, t_ref);
    tod_copy<3>(tt, p210, -1.0).perform(false, t_ref);
    tod_copy<3>(tt, p120, 1.0).perform(false, t_ref);
    tod_copy<3>(tt, p201, 1.0).perform(false, t_ref);
    ctf_tod_distribute<3>(t_ref).perform(dt);
    ctf_tod_collect<3>(dt).perform(t);

    compare_ref<3>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_distribute_test::test_5() {

    static const char testname[] = "ctf_tod_distribute_test::test_5()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> t(dims), tt(dims), t_ref(dims);
    sequence<4, unsigned> sym_grp(0), sym_sym(0);
    sym_grp[0] = 0; sym_grp[1] = 0; sym_grp[2] = 1; sym_grp[3] = 1;
    sym_sym[0] = 0; sym_sym[1] = 0;
    ctf_symmetry<4, double> sym(sym_grp, sym_sym);
    sym_sym[0] = 1; sym_sym[1] = 1;
    sym.add_component(sym_grp, sym_sym);
    ctf_dense_tensor<4, double> dt(dims, sym);

    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, t_ref);
    tod_copy<4>(tt, permutation<4>().permute(0, 1).permute(2, 3)).
        perform(false, t_ref);
    ctf_tod_distribute<4>(t_ref).perform(dt);
    ctf_tod_collect<4>(dt).perform(t);

    compare_ref<4>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

