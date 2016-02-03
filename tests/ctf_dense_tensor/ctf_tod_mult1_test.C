#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_mult1.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_mult1.h>
#include "../compare_ref.h"
#include "ctf_tod_mult1_test.h"

namespace libtensor {


void ctf_tod_mult1_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_mult_1a();
        test_mult_1b();
        test_div_1a();
        test_div_1b();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_mult1_test::test_mult_1a() {

    static const char testname[] = "ctf_tod_mult1_test::test_mult_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), ta_ref(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, ta_ref);
    tod_mult1<2>(tb, false, 0.5).perform(true, ta_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_mult1<2>(dtb, false, 0.5).perform(true, dta);
    ctf_tod_collect<2>(dta).perform(ta);

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult1_test::test_mult_1b() {

    static const char testname[] = "ctf_tod_mult1_test::test_mult_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), ta_ref(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, ta_ref);
    tod_mult1<2>(tb, false, 0.5).perform(false, ta_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_mult1<2>(dtb, false, 0.5).perform(false, dta);
    ctf_tod_collect<2>(dta).perform(ta);

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult1_test::test_div_1a() {

    static const char testname[] = "ctf_tod_mult1_test::test_div_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), ta_ref(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, ta_ref);
    tod_mult1<2>(tb, true, 0.5).perform(true, ta_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_mult1<2>(dtb, true, 0.5).perform(true, dta);
    ctf_tod_collect<2>(dta).perform(ta);

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult1_test::test_div_1b() {

    static const char testname[] = "ctf_tod_mult1_test::test_div_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), ta_ref(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, ta_ref);
    tod_mult1<2>(tb, true, 0.5).perform(false, ta_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_mult1<2>(dtb, true, 0.5).perform(false, dta);
    ctf_tod_collect<2>(dta).perform(ta);

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

