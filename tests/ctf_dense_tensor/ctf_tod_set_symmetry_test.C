#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set_symmetry.h>
#include "../compare_ref.h"
#include "ctf_tod_set_symmetry_test.h"

namespace libtensor {


void ctf_tod_set_symmetry_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();
        test_2();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_set_symmetry_test::test_1a() {

    static const char testname[] = "ctf_tod_set_symmetry_test::test_1a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2));
    sequence<2, unsigned> syma_grp, syma_sym;
    syma_grp[0] = 0; syma_grp[1] = 0; syma_sym[0] = 0;
    ctf_symmetry<2, double> syma1(syma_grp, syma_sym);
    syma_grp[0] = 0; syma_grp[1] = 1; syma_sym[0] = 0; syma_sym[1] = 0;
    ctf_symmetry<2, double> syma2(syma_grp, syma_sym);

    dense_tensor<2, double, allocator_t> ta(dimsa), ta_ref(dimsa);
    ctf_dense_tensor<2, double> dta(dimsa, syma1);

    ctf_tod_random<2>().perform(true, dta);
    ctf_tod_collect<2>(dta).perform(ta);
    tod_copy<2>(ta, permutation<2>().permute(0, 1)).perform(true, ta_ref);
    ctf_tod_set_symmetry<2>(syma2).perform(false, dta);
    ctf_tod_collect<2>(dta).perform(ta);

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_set_symmetry_test::test_1b() {

    static const char testname[] = "ctf_tod_set_symmetry_test::test_1b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2));
    sequence<2, unsigned> syma_grp, syma_sym;
    syma_grp[0] = 0; syma_grp[1] = 0; syma_sym[0] = 1;
    ctf_symmetry<2, double> syma1(syma_grp, syma_sym);
    syma_grp[0] = 0; syma_grp[1] = 1; syma_sym[0] = 0; syma_sym[1] = 0;
    ctf_symmetry<2, double> syma2(syma_grp, syma_sym);

    dense_tensor<2, double, allocator_t> ta(dimsa), ta_ref(dimsa);
    ctf_dense_tensor<2, double> dta(dimsa, syma1);

    ctf_tod_random<2>().perform(true, dta);
    ctf_tod_collect<2>(dta).perform(ta);
    tod_copy<2>(ta, permutation<2>().permute(0, 1), -1.0).perform(true, ta_ref);
    ctf_tod_set_symmetry<2>(syma2).perform(false, dta);
    ctf_tod_collect<2>(dta).perform(ta);

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_set_symmetry_test::test_2() {

    static const char testname[] = "ctf_tod_set_symmetry_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dimsa(index_range<4>(i1, i2));
    sequence<4, unsigned> syma_grp, syma_sym;
    syma_grp[0] = 0; syma_grp[1] = 0; syma_grp[2] = 1; syma_grp[3] = 1;
    syma_sym[0] = 0; syma_sym[1] = 0;
    ctf_symmetry<4, double> syma1(syma_grp, syma_sym);
    syma_grp[0] = 0; syma_grp[1] = 0; syma_grp[2] = 1; syma_grp[3] = 2;
    syma_sym[0] = 0; syma_sym[1] = 0; syma_sym[2] = 0;
    ctf_symmetry<4, double> syma2(syma_grp, syma_sym);

    dense_tensor<4, double, allocator_t> ta(dimsa), ta_ref(dimsa);
    ctf_dense_tensor<4, double> dta(dimsa, syma1);

    ctf_tod_random<4>().perform(true, dta);
    ctf_tod_collect<4>(dta).perform(ta);
    tod_copy<4>(ta, permutation<4>().permute(0, 1)).perform(true, ta_ref);
    ctf_tod_set_symmetry<4>(syma2).perform(false, dta);
    ctf_tod_collect<4>(dta).perform(ta);

    compare_ref<4>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

