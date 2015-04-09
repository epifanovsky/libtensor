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

    ctf::exit();
}


void ctf_tod_distribute_test::test_1() {

    static const char testname[] = "ctf_tod_distribute_test::test_1()";

    typedef std_allocator<double> allocator_t;

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

    typedef std_allocator<double> allocator_t;

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


} // namespace libtensor

